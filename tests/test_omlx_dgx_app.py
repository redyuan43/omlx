# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from pathlib import Path

import httpx

from omlx_dgx.config import DGXSettingsManager, ModelProfile
import omlx_dgx.control_plane.app as app_module
from omlx_dgx.control_plane.benchmarks import BenchmarkManager
from omlx_dgx.control_plane.app import create_app
from omlx_dgx.runtime.backend import BackendAdapter, BackendCapabilities, RuntimeMetrics
from omlx_dgx.tiered_kv import PersistentManifestStore


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data: dict | None = None,
        headers: dict | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = headers or {"content-type": "application/json"}
        self.text = text or json.dumps(self._json_data)

    def json(self):
        return self._json_data


class FakeBackend(BackendAdapter):
    def __init__(self, *, capabilities: BackendCapabilities | None = None) -> None:
        self.calls = []
        self._capabilities = capabilities or BackendCapabilities(
            chat_completions=True,
            completions=True,
        )

    def health(self) -> bool:
        return True

    def list_models(self) -> dict:
        return {"object": "list", "data": [{"id": "backend-model"}]}

    def proxy(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        payload = kwargs.get("json") or {}
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": payload.get("model"),
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "pong"},
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 2},
                }
            )
        if path == "v1/completions":
            return FakeResponse(
                json_data={
                    "id": "cmpl-test",
                    "object": "text_completion",
                    "model": payload.get("model"),
                    "choices": [{"text": "pong", "finish_reason": "stop"}],
                }
            )
        if path == "v1/embeddings":
            return FakeResponse(
                json_data={
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": payload.get("model"),
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                }
            )
        if path == "v1/rerank":
            return FakeResponse(
                json_data={
                    "id": "rerank-test",
                    "results": [
                        {
                            "index": 0,
                            "relevance_score": 0.9,
                            "document": {"text": "doc-a"},
                        }
                    ],
                    "model": payload.get("model"),
                    "usage": {"total_tokens": 12},
                }
            )
        return FakeResponse(json_data={"id": "generic", "model": payload.get("model")})

    def collect_metrics(self) -> RuntimeMetrics:
        return RuntimeMetrics(
            backend_url="http://fake-backend",
            healthy=True,
            gpu_name="Fake GPU",
            gpu_memory_used_mb=1,
            gpu_memory_total_mb=2,
            gpu_util_percent=3,
            gpu_temperature_c=4,
            details={
                "session_restore": {
                    "enabled": True,
                    "counts": {"restored": 1},
                }
            },
        )

    def hicache_storage_status(self) -> dict:
        return {"hicache_storage_backend": "file"}

    def attach_hicache_storage_backend(self, overrides=None) -> dict:
        return {"attached": True, "overrides": overrides or {}}

    def detach_hicache_storage_backend(self) -> dict:
        return {"detached": True}

    def cache_report(self) -> dict:
        return {"enable_cache_report": True, "internal_states": [{"cached_tokens": 12}]}

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities


class FakeModelPoolBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.models = {
            "primary": {
                "model_id": "primary",
                "loaded": True,
                "pinned": True,
                "ttl_seconds": 0,
                "last_used_at": 1.0,
                "last_unload_reason": "",
            }
        }
        self.default_model_id = "primary"

    def model_pool_diagnostics(self) -> dict:
        return {
            "enabled": True,
            "default_model_id": self.default_model_id,
            "max_loaded_models": 2,
            "loaded_models": [
                model_id for model_id, item in self.models.items() if item["loaded"]
            ],
            "models": [dict(item) for item in self.models.values()],
            "recent_events": [],
        }

    def register_model(self, registration, *, is_default: bool = False) -> dict:
        self.models[registration.model_id] = {
            "model_id": registration.model_id,
            "loaded": False,
            "pinned": registration.pinned,
            "ttl_seconds": registration.ttl_seconds,
            "last_used_at": None,
            "last_unload_reason": "",
        }
        if is_default:
            self.default_model_id = registration.model_id
        return self.model_pool_diagnostics()

    def load_model(self, model_id: str, *, reason: str = "manual_load") -> dict:
        self.models[model_id]["loaded"] = True
        self.models[model_id]["last_used_at"] = 2.0
        return {"model_id": model_id, "loaded": True, "reason": reason}

    def unload_model(self, model_id: str, *, reason: str = "manual_unload") -> dict:
        self.models[model_id]["loaded"] = False
        self.models[model_id]["last_unload_reason"] = reason
        return {"model_id": model_id, "loaded": False, "reason": reason}

    def set_model_pin(self, model_id: str, pinned: bool) -> dict:
        self.models[model_id]["pinned"] = pinned
        return {"model_id": model_id, "pinned": pinned}


def _request(app, method: str, path: str, **kwargs):
    async def _run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(_run())


def test_control_plane_rewrites_model_alias(tmp_path: Path):
    settings = DGXSettingsManager(tmp_path / "state")
    settings.ensure_model(
        ModelProfile(model_id="qwen35-35b", model_alias="qwen35", is_default=True)
    )
    store = PersistentManifestStore(tmp_path / "cache")
    app = create_app(
        settings_manager=settings,
        backend=FakeBackend(),
        manifest_store=store,
    )

    response = _request(
        app,
        "POST",
        "/v1/chat/completions",
        json={"model": "qwen35", "stream": False},
    )
    assert response.status_code == 200
    assert response.json()["model"] == "qwen35-35b"


def test_admin_runtime_endpoint_returns_metrics(tmp_path: Path):
    settings = DGXSettingsManager(tmp_path / "state")
    app = create_app(
        settings_manager=settings,
        backend=FakeBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(app, "GET", "/admin/api/runtime")
    assert response.status_code == 200
    assert response.json()["backend"]["gpu_name"] == "Fake GPU"
    assert response.json()["backend"]["details"]["session_restore"]["enabled"] is True
    assert response.json()["capabilities"]["services"]["/v1/messages"]["supported"] is True
    assert response.json()["hicache_storage"]["hicache_storage_backend"] == "file"
    assert response.json()["cache_report"]["enable_cache_report"] is True


def test_messages_endpoint_adapts_to_chat_completions(tmp_path: Path):
    backend = FakeBackend()
    app = create_app(
        settings_manager=DGXSettingsManager(tmp_path / "state"),
        backend=backend,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(
        app,
        "POST",
        "/v1/messages",
        json={
            "model": "backend-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "message"
    assert payload["content"][0]["type"] == "text"
    assert payload["content"][0]["text"] == "pong"
    assert backend.calls[-1][1] == "v1/chat/completions"
    assert backend.calls[-1][2]["json"]["messages"] == [{"role": "user", "content": "Hello"}]


def test_messages_endpoint_preserves_images_for_multimodal_backend(tmp_path: Path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(
            chat_completions=True,
            completions=True,
            vision_chat=True,
        )
    )
    settings = DGXSettingsManager(tmp_path / "state")
    settings.ensure_model(
        ModelProfile(
            model_id="vlm-model",
            model_alias="vlm",
            is_default=True,
            supports_vision=True,
        )
    )
    app = create_app(
        settings_manager=settings,
        backend=backend,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(
        app,
        "POST",
        "/v1/messages",
        json={
            "model": "vlm",
            "max_tokens": 32,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/a.png",
                            },
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    proxied = backend.calls[-1][2]["json"]
    assert proxied["model"] == "vlm-model"
    assert proxied["messages"][0]["content"][0]["type"] == "image_url"
    assert proxied["messages"][0]["content"][1]["type"] == "text"


def test_chat_completions_rejects_image_inputs_for_text_only_backend(tmp_path: Path):
    settings = DGXSettingsManager(tmp_path / "state")
    settings.ensure_model(ModelProfile(model_id="text-model", model_alias="text", is_default=True))
    app = create_app(
        settings_manager=settings,
        backend=FakeBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "text",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        },
                    ],
                }
            ],
        },
    )

    assert response.status_code == 501
    assert response.json()["detail"]["service"] == "vision_chat"


def test_chat_completions_rejects_ocr_requests_without_ocr_capability(tmp_path: Path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(
            chat_completions=True,
            completions=True,
            vision_chat=True,
        )
    )
    settings = DGXSettingsManager(tmp_path / "state")
    settings.ensure_model(
        ModelProfile(
            model_id="vlm-model",
            model_alias="vlm",
            is_default=True,
            supports_vision=True,
            supports_ocr=False,
        )
    )
    app = create_app(
        settings_manager=settings,
        backend=backend,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(
        app,
        "POST",
        "/v1/chat/completions",
        json={
            "model": "vlm",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the text in this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/a.png"},
                        },
                    ],
                }
            ],
            "metadata": {"omlx_task": "ocr"},
        },
    )

    assert response.status_code == 501
    assert response.json()["detail"]["service"] == "ocr"


def test_embeddings_endpoint_proxies_when_backend_declares_capability(tmp_path: Path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(
            chat_completions=True,
            completions=True,
            embeddings=True,
        )
    )
    app = create_app(
        settings_manager=DGXSettingsManager(tmp_path / "state"),
        backend=backend,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "embed-model", "input": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2]
    assert backend.calls[-1][1] == "v1/embeddings"


def test_rerank_endpoint_proxies_when_backend_declares_capability(tmp_path: Path):
    backend = FakeBackend(
        capabilities=BackendCapabilities(
            chat_completions=True,
            completions=True,
            rerank=True,
        )
    )
    app = create_app(
        settings_manager=DGXSettingsManager(tmp_path / "state"),
        backend=backend,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(
        app,
        "POST",
        "/v1/rerank",
        json={"model": "rerank-model", "query": "hello", "documents": ["doc-a"]},
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["relevance_score"] == 0.9
    assert backend.calls[-1][1] == "v1/rerank"


def test_unsupported_capability_responses_are_explicit(tmp_path: Path):
    app = create_app(
        settings_manager=DGXSettingsManager(tmp_path / "state"),
        backend=FakeBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    embeddings = _request(
        app,
        "POST",
        "/v1/embeddings",
        json={"model": "embed-model", "input": "hello"},
    )
    rerank = _request(
        app,
        "POST",
        "/v1/rerank",
        json={"model": "rerank-model", "query": "hello", "documents": ["doc-a"]},
    )

    assert embeddings.status_code == 501
    assert embeddings.json()["detail"]["type"] == "unsupported_capability"
    assert embeddings.json()["detail"]["service"] == "embeddings"
    assert rerank.status_code == 501
    assert rerank.json()["detail"]["service"] == "rerank"


def test_admin_hicache_endpoints_round_trip(tmp_path: Path):
    settings = DGXSettingsManager(tmp_path / "state")
    app = create_app(
        settings_manager=settings,
        backend=FakeBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(app, "GET", "/admin/api/runtime/hicache/storage-backend")
    assert response.status_code == 200
    assert response.json()["hicache_storage_backend"] == "file"

    response = _request(
        app,
        "PUT",
        "/admin/api/runtime/hicache/storage-backend",
        json={"hicache_storage_backend": "dynamic"},
    )
    assert response.status_code == 200
    assert response.json()["overrides"]["hicache_storage_backend"] == "dynamic"

    response = _request(app, "DELETE", "/admin/api/runtime/hicache/storage-backend")
    assert response.status_code == 200
    assert response.json()["detached"] is True

    response = _request(app, "GET", "/admin/api/runtime/cache-report")
    assert response.status_code == 200
    assert response.json()["cache_report"]["internal_states"][0]["cached_tokens"] == 12


def test_create_app_selects_llama_cpp_backend(tmp_path: Path, monkeypatch):
    settings = DGXSettingsManager(tmp_path / "state")
    settings.config.backend.kind = "llama_cpp"
    settings.config.backend.quant_mode = "gguf_experimental"
    settings.config.backend.model_source = "gguf"
    settings.config.backend.artifact_path = "/models/Qwen3.5-4B-Q4_K_M.gguf"
    settings.save()

    created = {}

    class _FactoryBackend(FakeBackend):
        pass

    def fake_from_runtime_config(config, state_dir):
        created["config"] = config
        created["state_dir"] = str(state_dir)
        return _FactoryBackend()

    monkeypatch.setattr(
        app_module.LlamaCppModelPoolAdapter,
        "from_runtime_config",
        staticmethod(fake_from_runtime_config),
    )

    app = create_app(
        settings_manager=settings,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(app, "GET", "/health")

    assert response.status_code == 200
    assert created["config"].backend.kind == "llama_cpp"
    assert created["config"].backend.artifact_path == "/models/Qwen3.5-4B-Q4_K_M.gguf"
    assert response.json()["capabilities"]["services"]["/admin/api/benchmarks"]["supported"] is True


def test_admin_model_pool_endpoints_round_trip(tmp_path: Path):
    settings = DGXSettingsManager(tmp_path / "state")
    settings.ensure_model(ModelProfile(model_id="primary", model_alias="qwen35", is_default=True))
    app = create_app(
        settings_manager=settings,
        backend=FakeModelPoolBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(app, "GET", "/admin/api/runtime/model-pool")
    assert response.status_code == 200
    assert response.json()["default_model_id"] == "primary"

    response = _request(
        app,
        "POST",
        "/admin/api/runtime/model-pool",
        json={
            "model_id": "secondary",
            "model_alias": "qwen35-secondary",
            "artifact_path": "/models/Qwen3.5-4B-Q4_K_S.gguf",
            "base_url": "http://127.0.0.1:32121",
            "gguf_variant": "Q4_K_S",
            "ttl_seconds": 600,
            "idle_unload_seconds": 120,
        },
    )
    assert response.status_code == 200
    assert any(model["model_id"] == "secondary" for model in response.json()["models"])

    response = _request(
        app,
        "POST",
        "/admin/api/runtime/model-pool/load",
        json={"model_id": "secondary"},
    )
    assert response.status_code == 200
    assert response.json()["reason"] == "manual_load"

    response = _request(
        app,
        "POST",
        "/admin/api/runtime/model-pool/pin",
        json={"model_id": "secondary", "pinned": True},
    )
    assert response.status_code == 200
    assert response.json()["pinned"] is True

    response = _request(
        app,
        "POST",
        "/admin/api/runtime/model-pool/unload",
        json={"model_id": "secondary"},
    )
    assert response.status_code == 200
    assert response.json()["reason"] == "manual_unload"

    reloaded = DGXSettingsManager(tmp_path / "state")
    registration = reloaded.config.backend.model_pool.models["secondary"]
    assert registration.base_url == "http://127.0.0.1:32121"
    assert registration.idle_unload_seconds == 120
    assert registration.pinned is True


def test_admin_benchmark_endpoints_run_and_retrieve_latest_report(tmp_path: Path, monkeypatch):
    manager = BenchmarkManager(tmp_path / "state")

    class _CompletedProcess:
        returncode = 0
        stdout = json.dumps(
            {
                "urls": {
                    "control_plane_url": "http://127.0.0.1:8010",
                    "runtime_url": "http://127.0.0.1:31000",
                },
                "runtime_summary": {"backend_format": "llama_cpp_gguf"},
                "single_session_followup": {"summary": {"followup_avg_sec": 0.123}},
            }
        )
        stderr = ""

    monkeypatch.setattr(
        "omlx_dgx.control_plane.benchmarks.subprocess.run",
        lambda *args, **kwargs: _CompletedProcess(),
    )

    app = create_app(
        settings_manager=DGXSettingsManager(tmp_path / "settings"),
        backend=FakeBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
        benchmark_manager=manager,
    )

    listed = _request(app, "GET", "/admin/api/benchmarks")
    assert listed.status_code == 200
    names = [item["name"] for item in listed.json()["benchmarks"]]
    assert "qwen35-4b" in names
    assert "multimodal-smoke" in names

    run_response = _request(
        app,
        "POST",
        "/admin/api/benchmarks/qwen35-4b/run",
        json={"control_plane_url": "http://127.0.0.1:8010"},
    )
    assert run_response.status_code == 200
    report_id = run_response.json()["report_id"]

    reports = _request(app, "GET", "/admin/api/benchmarks/qwen35-4b/reports")
    assert reports.status_code == 200
    assert reports.json()["reports"][0]["report_id"] == report_id

    latest = _request(app, "GET", "/admin/api/benchmarks/qwen35-4b/latest")
    assert latest.status_code == 200
    assert latest.json()["report_id"] == report_id
    assert latest.json()["report"]["single_session_followup"]["summary"]["followup_avg_sec"] == 0.123
