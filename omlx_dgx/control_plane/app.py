# SPDX-License-Identifier: Apache-2.0
"""FastAPI control-plane for DGX runtimes and tiered cache introspection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from omlx_dgx.config import DGXSettingsManager
from omlx_dgx.runtime import SGLangBackendAdapter, TensorRTLLMBackendAdapter
from omlx_dgx.runtime.backend import BackendAdapter, BackendError, HttpOpenAIBackendAdapter
from omlx_dgx.tiered_kv import PersistentManifestStore


@dataclass
class AppServices:
    settings: DGXSettingsManager
    backend: BackendAdapter
    manifest_store: PersistentManifestStore


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
    <script>
      async function load() {
        const runtime = await fetch('/admin/api/runtime').then(r => r.json());
        document.getElementById('runtime').textContent = JSON.stringify(runtime, null, 2);
        const records = await fetch('/admin/api/cache/records').then(r => r.json());
        document.getElementById('records').textContent = JSON.stringify(records, null, 2);
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


def create_app(
    *,
    base_path: str | Path = "~/.omlx-dgx",
    settings_manager: Optional[DGXSettingsManager] = None,
    backend: Optional[BackendAdapter] = None,
    manifest_store: Optional[PersistentManifestStore] = None,
) -> FastAPI:
    root_path = Path(base_path).expanduser().resolve()
    settings = settings_manager or DGXSettingsManager(root_path)
    config = settings.config
    store = manifest_store or PersistentManifestStore(
        Path(config.cache.ssd_root).expanduser().resolve()
    )
    if backend is not None:
        backend_adapter = backend
    elif config.backend.kind == "sglang":
        backend_adapter = SGLangBackendAdapter.from_backend_config(
            config.backend,
            root_path,
        )
    elif config.backend.kind == "tensorrt_llm":
        backend_adapter = TensorRTLLMBackendAdapter.from_backend_config(
            config.backend,
            root_path,
        )
    else:
        backend_adapter = HttpOpenAIBackendAdapter(config.backend.base_url)
    services = AppServices(settings=settings, backend=backend_adapter, manifest_store=store)

    app = FastAPI(title="oMLX DGX Control Plane")
    app.state.services = services

    @app.get("/health")
    async def health() -> dict:
        return {
            "ok": True,
            "backend_healthy": services.backend.health(),
            "models": services.settings.config.public_models(),
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
        return await _proxy_request(request, services, "v1/chat/completions")

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await _proxy_request(request, services, "v1/completions")

    @app.get("/admin", response_class=HTMLResponse)
    async def admin() -> str:
        return _render_admin()

    @app.get("/admin/api/runtime")
    async def runtime() -> dict:
        payload = {
            "config": services.settings.config.to_dict(),
            "backend": services.backend.collect_metrics().to_dict(),
            "cold_store": services.manifest_store.stats(),
        }
        for key, func in (
            ("hicache_storage", services.backend.hicache_storage_status),
            ("cache_report", services.backend.cache_report),
        ):
            try:
                payload[key] = func()
            except BackendError as exc:
                payload[f"{key}_error"] = str(exc)
        return payload

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
        content=response.json(),
    )
