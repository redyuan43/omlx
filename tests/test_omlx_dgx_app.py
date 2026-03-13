# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import httpx

from omlx_dgx.config import DGXSettingsManager, ModelProfile
import omlx_dgx.control_plane.app as app_module
from omlx_dgx.control_plane.app import create_app
from omlx_dgx.runtime.backend import BackendAdapter, RuntimeMetrics
from omlx_dgx.tiered_kv import PersistentManifestStore


class FakeBackend(BackendAdapter):
    def health(self) -> bool:
        return True

    def list_models(self) -> dict:
        return {"object": "list", "data": [{"id": "backend-model"}]}

    def proxy(self, method: str, path: str, **kwargs):
        class Response:
            status_code = 200

            @staticmethod
            def json():
                return {"id": "chatcmpl-test", "model": kwargs["json"]["model"]}

        return Response()

    def collect_metrics(self) -> RuntimeMetrics:
        return RuntimeMetrics(
            backend_url="http://fake-backend",
            healthy=True,
            gpu_name="Fake GPU",
            gpu_memory_used_mb=1,
            gpu_memory_total_mb=2,
            gpu_util_percent=3,
            gpu_temperature_c=4,
        )

    def hicache_storage_status(self) -> dict:
        return {"hicache_storage_backend": "file"}

    def attach_hicache_storage_backend(self, overrides=None) -> dict:
        return {"attached": True, "overrides": overrides or {}}

    def detach_hicache_storage_backend(self) -> dict:
        return {"detached": True}

    def cache_report(self) -> dict:
        return {"enable_cache_report": True, "internal_states": [{"cached_tokens": 12}]}


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
    assert response.json()["hicache_storage"]["hicache_storage_backend"] == "file"
    assert response.json()["cache_report"]["enable_cache_report"] is True


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

    def fake_from_backend_config(config, state_dir):
        created["config"] = config
        created["state_dir"] = str(state_dir)
        return _FactoryBackend()

    monkeypatch.setattr(
        app_module.LlamaCppBackendAdapter,
        "from_backend_config",
        staticmethod(fake_from_backend_config),
    )

    app = create_app(
        settings_manager=settings,
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )

    response = _request(app, "GET", "/health")

    assert response.status_code == 200
    assert created["config"].kind == "llama_cpp"
    assert created["config"].artifact_path == "/models/Qwen3.5-4B-Q4_K_M.gguf"
