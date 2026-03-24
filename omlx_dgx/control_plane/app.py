# SPDX-License-Identifier: Apache-2.0
"""FastAPI control-plane for DGX runtimes and tiered cache introspection."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import ValidationError

from omlx.api.anthropic_models import MessagesRequest
from omlx.api.anthropic_utils import (
    convert_anthropic_to_internal,
    convert_anthropic_tools_to_internal,
    convert_internal_to_anthropic_response,
)
from omlx.api.openai_models import ToolCall
from omlx_dgx.config import (
    DGXSettingsManager,
    LlamaCppModelRegistration,
    ModelProfile,
)
from omlx_dgx.control_plane.benchmarks import (
    BenchmarkExecutionError,
    BenchmarkManager,
)
from omlx_dgx.runtime import (
    AdaptiveBackendAdapter,
    SGLangBackendAdapter,
    TensorRTLLMBackendAdapter,
)
from omlx_dgx.runtime.backend import BackendAdapter, BackendError, HttpOpenAIBackendAdapter
from omlx_dgx.runtime.adaptive import derive_secondary_base_url
from omlx_dgx.runtime.llama_cpp import LlamaCppModelPoolAdapter
from omlx_dgx.tiered_kv import PersistentManifestStore


@dataclass
class AppServices:
    settings: DGXSettingsManager
    backend: BackendAdapter
    manifest_store: PersistentManifestStore
    benchmark_manager: BenchmarkManager


def _render_admin() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>oMLX DGX Control Plane</title>
    <style>
      :root { color-scheme: light; }
      body { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; margin: 2rem; background: #f5f1e8; color: #17202a; }
      .panel { background: #fffaf2; border: 1px solid #d7c7ac; border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1rem; box-shadow: 0 10px 25px rgba(23, 32, 42, 0.08); }
      pre { white-space: pre-wrap; word-break: break-word; }
      h1 { margin-bottom: 0.25rem; }
      .muted { color: #6b7280; }
    </style>
  </head>
  <body>
    <h1>oMLX DGX</h1>
    <p class="muted">DGX control plane with managed runtimes and HiCache visibility.</p>
    <div class="panel">
      <h2>Runtime</h2>
      <pre id="runtime">Loading…</pre>
    </div>
    <div class="panel">
      <h2>Cold Cache Records</h2>
      <pre id="records">Loading…</pre>
    </div>
    <div class="panel">
      <h2>Model Pool</h2>
      <pre id="model-pool">Loading…</pre>
    </div>
    <script>
      async function load() {
        const runtime = await fetch('/admin/api/runtime').then(r => r.json());
        document.getElementById('runtime').textContent = JSON.stringify(runtime, null, 2);
        const records = await fetch('/admin/api/cache/records').then(r => r.json());
        document.getElementById('records').textContent = JSON.stringify(records, null, 2);
        const modelPool = await fetch('/admin/api/runtime/model-pool').then(r => r.json());
        document.getElementById('model-pool').textContent = JSON.stringify(modelPool, null, 2);
      }
      load();
    </script>
  </body>
</html>
"""


def _rewrite_model(payload: dict, settings: DGXSettingsManager) -> dict:
    updated = dict(payload)
    model_id = settings.config.resolve_model_id(updated.get("model"))
    if model_id:
        updated["model"] = model_id
    return updated


def _model_pool_backend(backend: BackendAdapter):
    diagnostics = getattr(backend, "model_pool_diagnostics", None)
    if callable(diagnostics):
        return backend
    return None


def _named_context_backend(backend: BackendAdapter):
    required = (
        "list_named_contexts",
        "get_named_context",
        "delete_named_context",
        "restore_named_context",
    )
    if all(callable(getattr(backend, name, None)) for name in required):
        return backend
    return None


def _parse_model_pool_registration(
    body: dict,
    settings: DGXSettingsManager,
) -> tuple[ModelProfile, LlamaCppModelRegistration]:
    model_id = str(body.get("model_id") or body.get("id") or "").strip()
    if not model_id:
        raise HTTPException(status_code=400, detail="model_id is required")

    existing_profile = settings.config.models.get(model_id)
    existing_registration = settings.config.backend.model_pool.models.get(model_id)

    model_alias = body.get("model_alias")
    if model_alias is None and existing_profile is not None:
        model_alias = existing_profile.model_alias
    is_default = bool(body.get("default", False))
    if not is_default and existing_profile is not None:
        is_default = existing_profile.is_default

    profile = ModelProfile(
        model_id=model_id,
        model_alias=model_alias,
        max_context_window=int(
            body.get(
                "max_context_window",
                (existing_profile.max_context_window if existing_profile else 0)
                or body.get("ctx_size")
                or settings.config.backend.ctx_size
                or 65536,
            )
        ),
        max_tokens=int(
            body.get(
                "max_tokens",
                (existing_profile.max_tokens if existing_profile else 0) or 4096,
            )
        ),
        temperature=float(
            body.get(
                "temperature",
                (existing_profile.temperature if existing_profile else 0.7),
            )
        ),
        top_p=float(
            body.get(
                "top_p",
                (existing_profile.top_p if existing_profile else 0.95),
            )
        ),
        top_k=int(
            body.get(
                "top_k",
                (existing_profile.top_k if existing_profile else 0),
            )
        ),
        is_default=is_default,
        supports_embeddings=bool(
            body.get(
                "supports_embeddings",
                existing_profile.supports_embeddings if existing_profile else False,
            )
        ),
        supports_rerank=bool(
            body.get(
                "supports_rerank",
                existing_profile.supports_rerank if existing_profile else False,
            )
        ),
        supports_vision=bool(
            body.get(
                "supports_vision",
                existing_profile.supports_vision if existing_profile else False,
            )
        ),
        supports_ocr=bool(
            body.get(
                "supports_ocr",
                existing_profile.supports_ocr if existing_profile else False,
            )
        ),
        primary_service=str(
            body.get(
                "primary_service",
                existing_profile.primary_service if existing_profile else "",
            )
            or ""
        ),
    )

    pinned = body.get("pinned")
    if pinned is None:
        pinned = existing_registration.pinned if existing_registration else profile.is_default
    ttl_seconds = body.get("ttl_seconds")
    if ttl_seconds is None:
        ttl_seconds = (
            existing_registration.ttl_seconds
            if existing_registration is not None
            else settings.config.backend.model_pool.default_ttl_seconds
        )
    idle_unload_seconds = body.get("idle_unload_seconds")
    if idle_unload_seconds is None:
        idle_unload_seconds = (
            existing_registration.idle_unload_seconds
            if existing_registration is not None
            else settings.config.backend.model_pool.default_idle_unload_seconds
        )

    registration = LlamaCppModelRegistration(
        model_id=model_id,
        model_alias=profile.model_alias,
        backend_kind=str(
            body.get(
                "backend_kind",
                existing_registration.backend_kind if existing_registration else "llama_cpp",
            )
            or "llama_cpp"
        ),
        backend_model_name=str(
            body.get(
                "backend_model_name",
                existing_registration.backend_model_name if existing_registration else "",
            )
            or ""
        ),
        model_repo_id=str(
            body.get(
                "model_repo_id",
                existing_registration.model_repo_id if existing_registration else "",
            )
            or ""
        ),
        artifact_path=str(
            body.get(
                "artifact_path",
                existing_registration.artifact_path if existing_registration else "",
            )
            or ""
        ),
        mmproj_path=str(
            body.get(
                "mmproj_path",
                existing_registration.mmproj_path if existing_registration else "",
            )
            or ""
        ),
        gguf_variant=str(
            body.get(
                "gguf_variant",
                existing_registration.gguf_variant if existing_registration else "",
            )
            or ""
        ),
        base_url=str(
            body.get(
                "base_url",
                existing_registration.base_url if existing_registration else "",
            )
            or ""
        ),
        launcher_cmd=str(
            body.get(
                "launcher_cmd",
                existing_registration.launcher_cmd if existing_registration else "",
            )
            or ""
        ),
        serving_preset=str(
            body.get(
                "serving_preset",
                existing_registration.serving_preset if existing_registration else "",
            )
            or ""
        ),
        ctx_size=int(
            body.get(
                "ctx_size",
                existing_registration.ctx_size if existing_registration else 0,
            )
            or 0
        ),
        parallel_slots=int(
            body.get(
                "parallel_slots",
                existing_registration.parallel_slots if existing_registration else 0,
            )
            or 0
        ),
        pinned=bool(pinned),
        ttl_seconds=int(ttl_seconds or 0),
        idle_unload_seconds=int(idle_unload_seconds or 0),
        supports_embeddings=bool(
            body.get(
                "supports_embeddings",
                existing_registration.supports_embeddings if existing_registration else False,
            )
        ),
        supports_rerank=bool(
            body.get(
                "supports_rerank",
                existing_registration.supports_rerank if existing_registration else False,
            )
        ),
        supports_vision=bool(
            body.get(
                "supports_vision",
                existing_registration.supports_vision if existing_registration else False,
            )
        ),
        supports_ocr=bool(
            body.get(
                "supports_ocr",
                existing_registration.supports_ocr if existing_registration else False,
            )
        ),
        primary_service=str(
            body.get(
                "primary_service",
                existing_registration.primary_service if existing_registration else "",
            )
            or ""
        ),
    )

    if registration.backend_kind != "openai_compatible":
        if not registration.artifact_path and not registration.model_repo_id:
            if existing_registration is None:
                raise HTTPException(
                    status_code=400,
                    detail="artifact_path or model_repo_id is required for a new pool model",
                )
    return profile, registration


def _resolve_model_profile(
    payload: dict,
    settings: DGXSettingsManager,
) -> Optional[ModelProfile]:
    model_id = settings.config.resolve_model_id(payload.get("model"))
    if not model_id:
        return None
    return settings.config.models.get(model_id)


def _content_has_image(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    for item in value:
        if hasattr(item, "model_dump"):
            item = item.model_dump(exclude_none=True)
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "")
        if item_type in {"image_url", "input_image", "image"}:
            return True
        image_url = item.get("image_url")
        if isinstance(image_url, dict) and image_url.get("url"):
            return True
        source = item.get("source")
        if isinstance(source, dict) and (source.get("url") or source.get("data")):
            return True
    return False


def _payload_has_image_inputs(payload: dict) -> bool:
    messages = payload.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if _content_has_image(message.get("content")):
                return True
    prompt = payload.get("prompt")
    if isinstance(prompt, list) and _content_has_image(prompt):
        return True
    return False


def _ocr_requested(payload: dict) -> bool:
    if payload.get("ocr") is True:
        return True
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        if metadata.get("ocr") is True:
            return True
        task = str(metadata.get("omlx_task") or metadata.get("task") or "").strip().lower()
        if task == "ocr":
            return True
    return False


def _effective_multimodal_support(
    services: AppServices,
    payload: dict,
) -> dict[str, bool]:
    backend_caps = services.backend.capabilities()
    profile = _resolve_model_profile(payload, services.settings)
    supports_vision = backend_caps.vision_chat
    supports_ocr = backend_caps.ocr
    if profile is not None:
        if services.settings.config.backend.kind == "openai_compatible":
            supports_vision = supports_vision or profile.supports_vision
            supports_ocr = supports_ocr or profile.supports_ocr
        else:
            supports_vision = supports_vision and profile.supports_vision
            supports_ocr = supports_ocr and profile.supports_ocr
    return {"vision_chat": supports_vision, "ocr": supports_ocr}


def _profile_supports_service(
    profile: Optional[ModelProfile],
    service_name: str,
) -> bool:
    if profile is None:
        return True
    if service_name == "embeddings":
        return profile.supports_embeddings
    if service_name == "rerank":
        return profile.supports_rerank
    if service_name in {"chat_completions", "completions", "messages"}:
        if profile.primary_service in {"embeddings", "rerank"}:
            return profile.supports_vision or profile.supports_ocr
        return True
    return True


def _specialized_generation_rejection(
    profile: Optional[ModelProfile],
    payload: dict,
) -> tuple[str, str] | None:
    if profile is None:
        return None
    primary_service = str(profile.primary_service or "")
    has_images = _payload_has_image_inputs(payload)
    ocr_requested = _ocr_requested(payload)
    if primary_service == "ocr":
        if not has_images:
            return (
                "ocr",
                "the selected OCR model requires image inputs",
            )
        if not ocr_requested:
            return (
                "ocr",
                "the selected OCR model only supports OCR-mode image requests",
            )
    if primary_service == "vision_chat":
        if not has_images:
            return (
                "vision_chat",
                "the selected vision model requires image inputs",
            )
        if ocr_requested:
            return (
                "ocr",
                "the selected vision model is not configured as an OCR service",
            )
    return None


def _effective_service_support(
    services: AppServices,
    path: str,
    payload: Optional[dict] = None,
) -> dict[str, Any]:
    capabilities = _service_capabilities(services)
    service = dict(capabilities["services"].get(path, {}))
    if not service:
        return {"service": "", "supported": False}
    if payload is None:
        return service
    profile = _resolve_model_profile(payload, services.settings)
    service["model_id"] = None if profile is None else profile.model_id
    if not _profile_supports_service(profile, str(service.get("service") or "")):
        service["supported"] = False
        service_name = str(service.get("service") or "")
        if service_name == "embeddings":
            service["detail"] = "the selected model does not support embeddings"
        elif service_name == "rerank":
            service["detail"] = "the selected model does not support rerank"
        else:
            service["detail"] = "the selected model does not support this generation surface"
    return service


def _service_capabilities(services: AppServices) -> dict:
    backend_caps = services.backend.capabilities()
    services_map = {
        "/v1/chat/completions": {
            "service": "chat_completions",
            "supported": backend_caps.chat_completions,
            "mode": "proxy" if backend_caps.chat_completions else "unsupported",
            "detail": (
                "proxied directly to the configured backend"
                if backend_caps.chat_completions
                else "the configured backend does not support chat completions"
            ),
            "vision_inputs_supported": backend_caps.vision_chat,
            "ocr_mode_supported": backend_caps.ocr,
        },
        "/v1/completions": {
            "service": "completions",
            "supported": backend_caps.completions,
            "mode": "proxy" if backend_caps.completions else "unsupported",
            "detail": (
                "proxied directly to the configured backend"
                if backend_caps.completions
                else "the configured backend does not support text completions"
            ),
        },
        "/v1/messages": {
            "service": "messages",
            "supported": backend_caps.chat_completions,
            "mode": "anthropic_chat_adapter" if backend_caps.chat_completions else "unsupported",
            "detail": (
                "Anthropic Messages requests are adapted onto /v1/chat/completions"
                if backend_caps.chat_completions
                else "the configured backend does not provide a chat surface for message adaptation"
            ),
            "vision_inputs_supported": backend_caps.vision_chat,
            "ocr_mode_supported": backend_caps.ocr,
        },
        "/v1/embeddings": {
            "service": "embeddings",
            "supported": backend_caps.embeddings,
            "mode": "proxy" if backend_caps.embeddings else "unsupported",
            "detail": (
                "proxied directly to the configured backend"
                if backend_caps.embeddings
                else "the configured backend does not support embeddings"
            ),
        },
        "/v1/rerank": {
            "service": "rerank",
            "supported": backend_caps.rerank,
            "mode": "proxy" if backend_caps.rerank else "unsupported",
            "detail": (
                "proxied directly to the configured backend"
                if backend_caps.rerank
                else "the configured backend does not support rerank"
            ),
        },
        "/admin/api/benchmarks": {
            "service": "benchmark_reports",
            "supported": True,
            "mode": "admin_task",
            "detail": "stable benchmark report listing, execution, and retrieval",
        },
    }
    return {
        "backend": {
            "kind": services.settings.config.backend.kind,
            "adapter": services.backend.__class__.__name__,
            "capabilities": backend_caps.to_dict(),
        },
        "models": services.settings.config.public_models(),
        "services": services_map,
    }


def _raise_unsupported_capability(
    services: AppServices,
    path: str,
    *,
    payload: Optional[dict] = None,
    detail_override: Optional[str] = None,
    service_override: Optional[str] = None,
) -> None:
    capabilities = _service_capabilities(services)
    service = _effective_service_support(services, path, payload)
    raise HTTPException(
        status_code=501,
        detail={
            "type": "unsupported_capability",
            "path": path,
            "service": service_override or service.get("service", ""),
            "backend_kind": services.settings.config.backend.kind,
            "backend_adapter": services.backend.__class__.__name__,
            "message": detail_override or service.get("detail", "requested service is not supported"),
            "supported_services": [
                route
                for route, item in capabilities["services"].items()
                if item.get("supported")
            ],
        },
    )


def _require_service_support(services: AppServices, path: str) -> None:
    capabilities = _service_capabilities(services)
    service = capabilities["services"].get(path)
    if not service or not service.get("supported"):
        _raise_unsupported_capability(services, path)


def _require_payload_service_support(
    services: AppServices,
    path: str,
    payload: dict,
) -> None:
    service = _effective_service_support(services, path, payload)
    if not service.get("supported"):
        _raise_unsupported_capability(
            services,
            path,
            payload=payload,
            detail_override=service.get("detail"),
            service_override=service.get("service"),
        )
    service_name = str(service.get("service") or "")
    if service_name in {"chat_completions", "completions", "messages"}:
        profile = _resolve_model_profile(payload, services.settings)
        specialized_rejection = _specialized_generation_rejection(profile, payload)
        if specialized_rejection is not None:
            service_override, detail = specialized_rejection
            _raise_unsupported_capability(
                services,
                path,
                payload=payload,
                detail_override=detail,
                service_override=service_override,
            )


def _require_multimodal_support(services: AppServices, path: str, payload: dict) -> None:
    if not _payload_has_image_inputs(payload):
        return
    support = _effective_multimodal_support(services, payload)
    if _ocr_requested(payload):
        if not support["ocr"]:
            _raise_unsupported_capability(
                services,
                path,
                detail_override=(
                    "the configured backend/model does not support OCR image requests"
                ),
                service_override="ocr",
            )
        return
    if not support["vision_chat"]:
        _raise_unsupported_capability(
            services,
            path,
            detail_override=(
                "the configured backend/model does not support vision-language image inputs"
            ),
            service_override="vision_chat",
        )


def _backend_json_content(response) -> Any:
    headers = getattr(response, "headers", {}) or {}
    content_type = headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return {
        "status_code": response.status_code,
        "text": getattr(response, "text", "").strip(),
    }


def _anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    if tool_choice is None:
        return None
    if hasattr(tool_choice, "model_dump"):
        payload = tool_choice.model_dump(exclude_none=True)
    elif isinstance(tool_choice, dict):
        payload = dict(tool_choice)
    else:
        return None
    kind = str(payload.get("type") or "").strip()
    if kind == "auto":
        return "auto"
    if kind == "any":
        return "required"
    if kind == "tool" and payload.get("name"):
        return {
            "type": "function",
            "function": {"name": str(payload["name"])},
        }
    return None


def _coerce_message_text(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def create_app(
    *,
    base_path: str | Path = "~/.omlx-dgx",
    settings_manager: Optional[DGXSettingsManager] = None,
    backend: Optional[BackendAdapter] = None,
    manifest_store: Optional[PersistentManifestStore] = None,
    benchmark_manager: Optional[BenchmarkManager] = None,
) -> FastAPI:
    root_path = Path(base_path).expanduser().resolve()
    settings = settings_manager or DGXSettingsManager(root_path)
    config = settings.config
    store = manifest_store or PersistentManifestStore(
        Path(config.cache.ssd_root).expanduser().resolve()
    )

    def build_sglang_backend() -> BackendAdapter:
        backend_config = config.backend
        if backend_config.prefill_strategy == "adaptive":
            primary_config = replace(
                backend_config,
                chunked_prefill_size=backend_config.adaptive_long_context_chunk_size,
            )
            secondary_config = replace(
                backend_config,
                base_url=backend_config.adaptive_backend_base_url
                or derive_secondary_base_url(backend_config.base_url),
                chunked_prefill_size=backend_config.adaptive_repeat_prefix_chunk_size,
            )
            primary = SGLangBackendAdapter.from_backend_config(
                primary_config,
                root_path / "adaptive" / "primary",
            )
            secondary = SGLangBackendAdapter.from_backend_config(
                secondary_config,
                root_path / "adaptive" / "secondary",
            )
            return AdaptiveBackendAdapter(
                primary=primary,
                secondary=secondary,
                primary_url=primary_config.base_url,
                secondary_url=secondary_config.base_url,
                primary_chunked_prefill_size=primary_config.chunked_prefill_size,
                secondary_chunked_prefill_size=secondary_config.chunked_prefill_size,
                short_prompt_threshold=backend_config.adaptive_short_prompt_threshold,
                max_sticky_sessions=backend_config.adaptive_max_sticky_sessions,
            )

        fixed_config = replace(
            backend_config,
            chunked_prefill_size=backend_config.fixed_chunked_prefill_size,
        )
        return SGLangBackendAdapter.from_backend_config(
            fixed_config,
            root_path,
        )

    if backend is not None:
        backend_adapter = backend
    elif config.backend.kind == "sglang":
        backend_adapter = build_sglang_backend()
    elif config.backend.kind == "tensorrt_llm":
        backend_adapter = TensorRTLLMBackendAdapter.from_backend_config(
            config.backend,
            root_path,
        )
    elif config.backend.kind == "llama_cpp":
        backend_adapter = LlamaCppModelPoolAdapter.from_runtime_config(
            config,
            root_path,
        )
    else:
        backend_adapter = HttpOpenAIBackendAdapter(config.backend.base_url)
    services = AppServices(
        settings=settings,
        backend=backend_adapter,
        manifest_store=store,
        benchmark_manager=benchmark_manager or BenchmarkManager(root_path),
    )

    app = FastAPI(title="oMLX DGX Control Plane")
    app.state.services = services

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "backend_healthy": services.backend.health(),
            "models": services.settings.config.public_models(),
            "capabilities": _service_capabilities(services),
        }

    @app.get("/v1/models")
    async def models() -> dict:
        configured = services.settings.config.public_models()
        if configured:
            return {"object": "list", "data": configured}
        try:
            return services.backend.list_models()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        _require_service_support(services, "/v1/chat/completions")
        return await _proxy_request(request, services, "v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        _require_service_support(services, "/v1/completions")
        return await _proxy_request(request, services, "v1/completions")

    @app.post("/v1/messages")
    async def messages(body: dict):
        _require_service_support(services, "/v1/messages")
        try:
            request_model = MessagesRequest.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        if request_model.stream:
            raise HTTPException(
                status_code=501,
                detail={
                    "type": "unsupported_capability",
                    "path": "/v1/messages",
                    "service": "messages_stream",
                    "backend_kind": services.settings.config.backend.kind,
                    "backend_adapter": services.backend.__class__.__name__,
                    "message": "Anthropic streaming responses are not supported on the DGX control plane yet",
                },
            )

        payload: dict[str, Any] = {
            "model": services.settings.config.resolve_model_id(request_model.model)
            or request_model.model,
            "messages": convert_anthropic_to_internal(
                request_model,
                preserve_images=any(
                    _content_has_image(message.content)
                    for message in request_model.messages
                ),
            ),
            "max_tokens": request_model.max_tokens,
            "stream": False,
        }
        if request_model.temperature is not None:
            payload["temperature"] = request_model.temperature
        if request_model.top_p is not None:
            payload["top_p"] = request_model.top_p
        if request_model.top_k is not None:
            payload["top_k"] = request_model.top_k
        if request_model.stop_sequences:
            payload["stop"] = request_model.stop_sequences
        if request_model.metadata:
            payload["metadata"] = request_model.metadata
        tools = convert_anthropic_tools_to_internal(request_model.tools)
        if tools:
            payload["tools"] = tools
        tool_choice = _anthropic_tool_choice_to_openai(request_model.tool_choice)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        chat_template_kwargs = dict(request_model.chat_template_kwargs or {})
        if request_model.thinking is not None:
            if request_model.thinking.type == "disabled":
                chat_template_kwargs.setdefault("enable_thinking", False)
            elif request_model.thinking.type == "enabled":
                chat_template_kwargs.setdefault("enable_thinking", True)
            if request_model.thinking.budget_tokens is not None:
                payload["reasoning_budget"] = request_model.thinking.budget_tokens
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        _require_payload_service_support(services, "/v1/messages", payload)
        _require_multimodal_support(services, "/v1/messages", payload)

        try:
            response = services.backend.proxy(
                "POST",
                "v1/chat/completions",
                json=payload,
                headers={"content-type": "application/json"},
            )
        except BackendError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        response_payload = _backend_json_content(response)
        if response.status_code >= 400:
            return JSONResponse(status_code=response.status_code, content=response_payload)

        choices = response_payload.get("choices") or []
        choice = choices[0] if choices else {}
        assistant_message = (
            choice.get("message")
            if isinstance(choice, dict)
            else {}
        )
        usage = response_payload.get("usage") or {}
        tool_calls = []
        raw_tool_calls = (
            assistant_message.get("tool_calls")
            if isinstance(assistant_message, dict)
            else None
        )
        if isinstance(raw_tool_calls, list):
            for item in raw_tool_calls:
                try:
                    tool_calls.append(ToolCall.model_validate(item))
                except ValidationError:
                    continue
        anthropic_response = convert_internal_to_anthropic_response(
            text=_coerce_message_text(assistant_message or {}),
            model=str(response_payload.get("model") or payload["model"]),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            finish_reason=choice.get("finish_reason") if isinstance(choice, dict) else None,
            tool_calls=tool_calls or None,
        )
        return JSONResponse(
            status_code=response.status_code,
            content=anthropic_response.model_dump(mode="json"),
        )

    @app.post("/v1/embeddings")
    async def embeddings(request: Request):
        _require_service_support(services, "/v1/embeddings")
        return await _proxy_request(request, services, "v1/embeddings")

    @app.post("/v1/rerank")
    async def rerank(request: Request):
        _require_service_support(services, "/v1/rerank")
        return await _proxy_request(request, services, "v1/rerank")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin() -> str:
        return _render_admin()

    @app.get("/admin/api/runtime")
    async def runtime() -> dict:
        payload = {
            "config": services.settings.config.to_dict(),
            "backend": services.backend.collect_metrics().to_dict(),
            "cold_store": services.manifest_store.stats(),
            "capabilities": _service_capabilities(services),
            "benchmark_summaries": services.benchmark_manager.latest_summaries(),
        }
        for key, func in (
            ("hicache_storage", services.backend.hicache_storage_status),
            ("cache_report", services.backend.cache_report),
        ):
            try:
                payload[key] = func()
            except BackendError as exc:
                payload[f"{key}_error"] = str(exc)
        pool_backend = _model_pool_backend(services.backend)
        if pool_backend is not None:
            payload["model_pool"] = pool_backend.model_pool_diagnostics()
        context_backend = _named_context_backend(services.backend)
        if context_backend is not None:
            try:
                payload["named_contexts"] = context_backend.list_named_contexts()
            except BackendError as exc:
                payload["named_contexts_error"] = str(exc)
        return payload

    @app.get("/admin/api/runtime/capabilities")
    async def runtime_capabilities() -> dict:
        return _service_capabilities(services)

    @app.get("/admin/api/runtime/model-pool")
    async def runtime_model_pool() -> dict:
        pool_backend = _model_pool_backend(services.backend)
        if pool_backend is None:
            raise HTTPException(status_code=400, detail="model pool is not supported by this backend")
        return pool_backend.model_pool_diagnostics()

    @app.post("/admin/api/runtime/model-pool")
    async def runtime_model_pool_register(request: Request) -> dict:
        pool_backend = _model_pool_backend(services.backend)
        if pool_backend is None:
            raise HTTPException(status_code=400, detail="model pool is not supported by this backend")
        body = json.loads((await request.body()) or b"{}")
        profile, registration = _parse_model_pool_registration(body, services.settings)
        services.settings.ensure_llama_cpp_model_registration(
            registration,
            profile=profile,
        )
        pool_backend.register_model(
            registration,
            is_default=profile.is_default,
        )
        return pool_backend.model_pool_diagnostics()

    @app.post("/admin/api/runtime/model-pool/load")
    async def runtime_model_pool_load(request: Request) -> dict:
        pool_backend = _model_pool_backend(services.backend)
        if pool_backend is None:
            raise HTTPException(status_code=400, detail="model pool is not supported by this backend")
        body = json.loads((await request.body()) or b"{}")
        model_id = str(body.get("model_id") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        try:
            return pool_backend.load_model(model_id, reason="manual_load")
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/runtime/model-pool/unload")
    async def runtime_model_pool_unload(request: Request) -> dict:
        pool_backend = _model_pool_backend(services.backend)
        if pool_backend is None:
            raise HTTPException(status_code=400, detail="model pool is not supported by this backend")
        body = json.loads((await request.body()) or b"{}")
        model_id = str(body.get("model_id") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        try:
            return pool_backend.unload_model(model_id, reason="manual_unload")
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/runtime/model-pool/pin")
    async def runtime_model_pool_pin(request: Request) -> dict:
        pool_backend = _model_pool_backend(services.backend)
        if pool_backend is None:
            raise HTTPException(status_code=400, detail="model pool is not supported by this backend")
        body = json.loads((await request.body()) or b"{}")
        model_id = str(body.get("model_id") or "").strip()
        if not model_id:
            raise HTTPException(status_code=400, detail="model_id is required")
        pinned = bool(body.get("pinned", True))
        registration = services.settings.config.backend.model_pool.models.get(model_id)
        if registration is None:
            raise HTTPException(status_code=404, detail=f"unknown model: {model_id}")
        registration.pinned = pinned
        services.settings.save()
        try:
            return pool_backend.set_model_pin(model_id, pinned)
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/contexts")
    async def admin_list_contexts(model: Optional[str] = None) -> dict:
        context_backend = _named_context_backend(services.backend)
        if context_backend is None:
            raise HTTPException(status_code=400, detail="named contexts are not supported by this backend")
        try:
            pool_backend = _model_pool_backend(services.backend)
            if pool_backend is not None:
                return context_backend.list_named_contexts(model_id=model)
            return context_backend.list_named_contexts()
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/contexts/{context_id}")
    async def admin_get_context(context_id: str, model: Optional[str] = None) -> dict:
        context_backend = _named_context_backend(services.backend)
        if context_backend is None:
            raise HTTPException(status_code=400, detail="named contexts are not supported by this backend")
        try:
            pool_backend = _model_pool_backend(services.backend)
            if pool_backend is not None:
                return context_backend.get_named_context(context_id, model_id=model)
            return context_backend.get_named_context(context_id)
        except BackendError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.delete("/admin/api/contexts/{context_id}")
    async def admin_delete_context(context_id: str, model: Optional[str] = None) -> dict:
        context_backend = _named_context_backend(services.backend)
        if context_backend is None:
            raise HTTPException(status_code=400, detail="named contexts are not supported by this backend")
        try:
            pool_backend = _model_pool_backend(services.backend)
            if pool_backend is not None:
                return context_backend.delete_named_context(context_id, model_id=model)
            return context_backend.delete_named_context(context_id)
        except BackendError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/admin/api/contexts/{context_id}/restore")
    async def admin_restore_context(context_id: str, model: Optional[str] = None) -> dict:
        context_backend = _named_context_backend(services.backend)
        if context_backend is None:
            raise HTTPException(status_code=400, detail="named contexts are not supported by this backend")
        try:
            pool_backend = _model_pool_backend(services.backend)
            if pool_backend is not None:
                return context_backend.restore_named_context(context_id, model_id=model)
            return context_backend.restore_named_context(context_id)
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/runtime/cache-report")
    async def runtime_cache_report() -> dict:
        try:
            return {
                "cache_report": services.backend.cache_report(),
            }
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/runtime/start")
    async def runtime_start() -> dict:
        try:
            return services.backend.start_runtime()
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/admin/api/runtime/stop")
    async def runtime_stop() -> dict:
        try:
            return services.backend.stop_runtime()
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/runtime/logs")
    async def runtime_logs(lines: int = 40) -> dict:
        try:
            return services.backend.runtime_logs(lines=lines)
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/runtime/hicache/storage-backend")
    async def hicache_storage_status() -> dict:
        try:
            return services.backend.hicache_storage_status()
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.put("/admin/api/runtime/hicache/storage-backend")
    async def hicache_storage_attach(request: Request) -> dict:
        try:
            body = await request.body()
            overrides = json.loads(body) if body else None
            return services.backend.attach_hicache_storage_backend(overrides=overrides)
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.delete("/admin/api/runtime/hicache/storage-backend")
    async def hicache_storage_detach() -> dict:
        try:
            return services.backend.detach_hicache_storage_backend()
        except BackendError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/admin/api/cache/records")
    async def cache_records(model_id: Optional[str] = None, limit: int = 50) -> dict:
        records = [
            record.__dict__
            for record in services.manifest_store.list_records(model_id=model_id, limit=limit)
        ]
        return {"records": records}

    @app.get("/admin/api/benchmarks")
    async def benchmarks() -> dict:
        return {
            "benchmarks": services.benchmark_manager.available_benchmarks(),
        }

    @app.get("/admin/api/benchmarks/{benchmark_name}/reports")
    async def benchmark_reports(benchmark_name: str, limit: int = 10) -> dict:
        try:
            reports = services.benchmark_manager.list_reports(benchmark_name, limit=limit)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "benchmark": benchmark_name,
            "reports": reports,
        }

    @app.get("/admin/api/benchmarks/{benchmark_name}/latest")
    async def benchmark_latest(benchmark_name: str) -> dict:
        try:
            report = services.benchmark_manager.latest_report(benchmark_name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"no benchmark report available for {benchmark_name}",
            )
        return report

    @app.post("/admin/api/benchmarks/{benchmark_name}/run")
    async def benchmark_run(benchmark_name: str, request: Request) -> dict:
        body = json.loads((await request.body()) or b"{}")
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="benchmark request body must be an object")
        try:
            return await asyncio.to_thread(
                services.benchmark_manager.run_benchmark,
                benchmark_name,
                overrides=body,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except BenchmarkExecutionError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


async def _proxy_request(request: Request, services: AppServices, path: str):
    body = await request.body()
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"host", "content-length"}
    }

    payload = None
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type and body:
        payload = _rewrite_model(json.loads(body), services.settings)
        _require_payload_service_support(services, f"/{path.lstrip('/')}", payload)
        _require_multimodal_support(services, f"/{path.lstrip('/')}", payload)
        body = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"

    stream = bool(payload and payload.get("stream"))

    try:
        response = services.backend.proxy(
            request.method,
            path,
            params=dict(request.query_params),
            data=body if payload is None else None,
            json=payload,
            headers=headers,
            stream=stream,
        )
    except BackendError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if stream:
        media_type = response.headers.get("content-type", "text/event-stream")
        return StreamingResponse(
            response.iter_content(chunk_size=8192),
            media_type=media_type,
            status_code=response.status_code,
        )

    return JSONResponse(
        status_code=response.status_code,
        content=_backend_json_content(response),
    )
