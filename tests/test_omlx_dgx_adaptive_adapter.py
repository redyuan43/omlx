# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from omlx_dgx.runtime.adaptive import AdaptiveBackendAdapter, derive_secondary_base_url
from omlx_dgx.runtime.backend import BackendAdapter, RuntimeMetrics


@dataclass
class _FakeResponse:
    status_code: int = 200

    def json(self):
        return {"ok": True}


class _FakeBackend(BackendAdapter):
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = []

    def health(self) -> bool:
        return True

    def list_models(self) -> dict:
        return {"data": [{"id": self.name}]}

    def proxy(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        return _FakeResponse()

    def collect_metrics(self) -> RuntimeMetrics:
        return RuntimeMetrics(
            backend_url=f"http://{self.name}",
            healthy=True,
            details={"backend_name": self.name},
        )

    def start_runtime(self) -> dict:
        return {"started": True, "backend": self.name}

    def stop_runtime(self) -> dict:
        return {"stopped": True, "backend": self.name}

    def runtime_logs(self, lines: int = 40) -> dict:
        return {"lines": [f"{self.name}-log"], "lines_requested": lines}


def test_derive_secondary_base_url_increments_port():
    assert derive_secondary_base_url("http://127.0.0.1:31000") == "http://127.0.0.1:31001"


def test_adaptive_backend_routes_long_and_sticky_sessions():
    primary = _FakeBackend("primary")
    secondary = _FakeBackend("secondary")
    adapter = AdaptiveBackendAdapter(
        primary=primary,
        secondary=secondary,
        primary_url="http://127.0.0.1:31000",
        secondary_url="http://127.0.0.1:31001",
        primary_chunked_prefill_size=8192,
        secondary_chunked_prefill_size=1024,
        short_prompt_threshold=100,
        max_sticky_sessions=8,
    )

    short_payload = {
        "user": "session-short",
        "messages": [{"role": "user", "content": "Reply with exactly PONG."}],
    }
    long_payload = {
        "user": "session-long",
        "messages": [
            {"role": "user", "content": "cache benchmark " * 80},
        ],
    }
    sticky_follow_up = {
        "user": "session-long",
        "messages": [{"role": "user", "content": "Short follow-up"}],
    }

    adapter.proxy("POST", "v1/chat/completions", json=short_payload)
    adapter.proxy("POST", "v1/chat/completions", json=long_payload)
    adapter.proxy("POST", "v1/chat/completions", json=sticky_follow_up)

    assert len(primary.calls) == 1
    assert len(secondary.calls) == 2

    metrics = adapter.collect_metrics().to_dict()
    routing = metrics["details"]["routing"]
    assert routing["profiles"]["primary"]["chunked_prefill_size"] == 8192
    assert routing["profiles"]["secondary"]["chunked_prefill_size"] == 1024
    assert routing["route_counts"]["short_prompt"] == 1
    assert routing["route_counts"]["long_prompt"] == 1
    assert routing["route_counts"]["sticky_session"] == 1
    assert routing["last_decision"]["backend_label"] == "secondary"
    assert routing["last_decision"]["reason"] == "sticky_session"


def test_adaptive_backend_lifecycle_is_aggregated():
    primary = _FakeBackend("primary")
    secondary = _FakeBackend("secondary")
    adapter = AdaptiveBackendAdapter(
        primary=primary,
        secondary=secondary,
        primary_url="http://127.0.0.1:31000",
        secondary_url="http://127.0.0.1:31001",
        primary_chunked_prefill_size=8192,
        secondary_chunked_prefill_size=1024,
        short_prompt_threshold=100,
    )

    started = adapter.start_runtime()
    logs = adapter.runtime_logs(lines=5)
    stopped = adapter.stop_runtime()

    assert started["mode"] == "adaptive"
    assert started["backends"]["primary"]["started"] is True
    assert started["backends"]["secondary"]["started"] is True
    assert logs["backends"]["primary"]["lines"] == ["primary-log"]
    assert logs["backends"]["secondary"]["lines"] == ["secondary-log"]
    assert stopped["backends"]["primary"]["stopped"] is True
    assert stopped["backends"]["secondary"]["stopped"] is True
