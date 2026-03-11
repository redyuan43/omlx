# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from fastapi.testclient import TestClient

from omlx_dgx.config import DGXSettingsManager, ModelProfile
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
    client = TestClient(app)

    response = client.post("/v1/chat/completions", json={"model": "qwen35", "stream": False})
    assert response.status_code == 200
    assert response.json()["model"] == "qwen35-35b"


def test_admin_runtime_endpoint_returns_metrics(tmp_path: Path):
    settings = DGXSettingsManager(tmp_path / "state")
    app = create_app(
        settings_manager=settings,
        backend=FakeBackend(),
        manifest_store=PersistentManifestStore(tmp_path / "cache"),
    )
    client = TestClient(app)

    response = client.get("/admin/api/runtime")
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
    client = TestClient(app)

    response = client.get("/admin/api/runtime/hicache/storage-backend")
    assert response.status_code == 200
    assert response.json()["hicache_storage_backend"] == "file"

    response = client.put(
        "/admin/api/runtime/hicache/storage-backend",
        json={"hicache_storage_backend": "dynamic"},
    )
    assert response.status_code == 200
    assert response.json()["overrides"]["hicache_storage_backend"] == "dynamic"

    response = client.delete("/admin/api/runtime/hicache/storage-backend")
    assert response.status_code == 200
    assert response.json()["detached"] is True

    response = client.get("/admin/api/runtime/cache-report")
    assert response.status_code == 200
    assert response.json()["cache_report"]["internal_states"][0]["cached_tokens"] == 12
