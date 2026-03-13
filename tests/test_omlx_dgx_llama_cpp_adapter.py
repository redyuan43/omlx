# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omlx_dgx.config import BackendConfig
from omlx_dgx.runtime.llama_cpp import LlamaCppBackendAdapter


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        json_data=None,
        text: str = "",
        headers=None,
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data
        self.text = text
        self.headers = headers or {}
        self.ok = 200 <= status_code < 400

    def json(self):
        return self._json_data


def test_llama_cpp_adapter_builds_local_gguf_launch_command(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32100",
        quant_mode="gguf_experimental",
        model_source="gguf",
        artifact_path=str(gguf_path),
        launcher_binary="/opt/llama.cpp/bin/llama-server",
        ctx_size=16384,
        parallel_slots=2,
        n_gpu_layers=999,
        flash_attn=True,
        batch_size=4096,
        ubatch_size=1024,
        cache_ram_mib=16384,
        cache_reuse=256,
        checkpoint_every_n_tokens=1024,
        ctx_checkpoints=64,
        slot_prompt_similarity=0.25,
        enable_runtime_metrics=True,
        split_mode="row",
        no_context_shift=True,
        jinja=True,
        reasoning_format="deepseek",
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert command[:3] == [
        "/opt/llama.cpp/bin/llama-server",
        "--model",
        str(gguf_path),
    ]
    assert "--ctx-size" in command
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert "--parallel" in command
    assert command[command.index("--parallel") + 1] == "2"
    assert "--n-gpu-layers" in command
    assert command[command.index("--n-gpu-layers") + 1] == "999"
    assert "--flash-attn" in command
    assert command[command.index("--flash-attn") + 1] == "on"
    assert "--batch-size" in command
    assert command[command.index("--batch-size") + 1] == "4096"
    assert "--ubatch-size" in command
    assert command[command.index("--ubatch-size") + 1] == "1024"
    assert "--cache-ram" in command
    assert command[command.index("--cache-ram") + 1] == "16384"
    assert "--cache-reuse" in command
    assert command[command.index("--cache-reuse") + 1] == "256"
    assert "--checkpoint-every-n-tokens" in command
    assert command[command.index("--checkpoint-every-n-tokens") + 1] == "1024"
    assert "--ctx-checkpoints" in command
    assert command[command.index("--ctx-checkpoints") + 1] == "64"
    assert "--slot-prompt-similarity" in command
    assert command[command.index("--slot-prompt-similarity") + 1] == "0.25"
    assert "--metrics" in command
    assert "--split-mode" in command
    assert command[command.index("--split-mode") + 1] == "row"
    assert "--no-context-shift" in command
    assert "--jinja" in command
    assert "--reasoning-format" in command
    assert command[command.index("--reasoning-format") + 1] == "deepseek"
    assert diagnostics["adapter"] == "llama_cpp"
    assert diagnostics["backend_format"] == "llama_cpp_gguf"
    assert diagnostics["quant_mode"] == "gguf_experimental"
    assert diagnostics["artifact_summary"]["artifact_kind"] == "local_file"
    assert diagnostics["gguf_variant"] == "Q4_K_M"
    assert diagnostics["batch_size"] == 4096
    assert diagnostics["cache_reuse"] == 256
    assert diagnostics["enable_session_stickiness"] is True
    assert diagnostics["sticky_session_prompt_threshold"] == 2048


def test_llama_cpp_adapter_uses_hf_flag_for_remote_gguf_reference(tmp_path: Path):
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32101",
        quant_mode="gguf_experimental",
        model_source="gguf",
        model_repo_id="lmstudio-community/Qwen3.5-4B-GGUF:Q4_K_S",
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert command[:3] == [
        "llama-server",
        "-hf",
        "lmstudio-community/Qwen3.5-4B-GGUF:Q4_K_S",
    ]
    assert diagnostics["model_flag"] == "-hf"
    assert diagnostics["artifact_summary"]["artifact_kind"] == "remote_ref"
    assert diagnostics["gguf_variant"] == "Q4_K_S"


def test_llama_cpp_adapter_collects_props_and_slots(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q6_K.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32102",
        quant_mode="gguf_experimental",
        model_source="gguf",
        artifact_path=str(gguf_path),
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    def fake_request(method, path, timeout=0, **kwargs):
        if path == "health":
            return FakeResponse(status_code=404, text="not found")
        if path == "v1/models":
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path == "props":
            return FakeResponse(
                json_data={
                    "model_path": str(gguf_path),
                    "default_generation_settings": {"n_ctx": 16384},
                },
                headers={"content-type": "application/json"},
            )
        if path == "slots":
            return FakeResponse(
                json_data={"slots": [{"id": 0, "state": "idle"}]},
                headers={"content-type": "application/json"},
            )
        if path == "metrics":
            return FakeResponse(
                text="# HELP llama_tokens tokens\nllama_tokens 42\n",
                headers={"content-type": "text/plain"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    metrics = adapter.collect_metrics().to_dict()
    cache_report = adapter.cache_report()

    assert metrics["healthy"] is True
    assert metrics["details"]["props"]["default_generation_settings"]["n_ctx"] == 16384
    assert metrics["details"]["slots"]["slots"][0]["state"] == "idle"
    assert metrics["details"]["slot_router"]["slot_summary"][0]["id"] == 0
    assert metrics["details"]["metrics_excerpt"][0] == "# HELP llama_tokens tokens"
    assert cache_report["props"]["model_path"] == str(gguf_path)
    assert cache_report["slots"]["slots"][0]["id"] == 0
    assert cache_report["slot_router"]["slot_summary"][0]["id"] == 0


def test_llama_cpp_adapter_uses_explicit_launcher_cmd(tmp_path: Path):
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32103",
        launcher_cmd="/opt/llama.cpp/bin/llama-server --model /tmp/model.gguf --port 30000",
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert command == [
        "/opt/llama.cpp/bin/llama-server",
        "--model",
        "/tmp/model.gguf",
        "--port",
        "30000",
    ]
    assert diagnostics["launcher_cmd"].startswith("/opt/llama.cpp/bin/llama-server")


def test_llama_cpp_adapter_assigns_sticky_slot_for_long_session(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32104",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=2,
        sticky_session_prompt_threshold=8,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []

    def fake_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path == "slots":
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "props":
            return FakeResponse(
                json_data={"default_generation_settings": {"n_ctx": 16384}},
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={"object": "chat.completion", "choices": [{"message": {"content": "OK"}}]},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    payload = {
        "model": "qwen35-4b-gguf",
        "messages": [{"role": "user", "content": "prefix " * 64}],
        "metadata": {"conversation_id": "session-a"},
    }
    adapter.proxy("POST", "v1/chat/completions", json=payload)
    adapter.proxy("POST", "v1/chat/completions", json=payload)

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    first_payload = request_calls[0][2]["json"]
    second_payload = request_calls[1][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert first_payload["id_slot"] == 0
    assert first_payload["cache_prompt"] is True
    assert second_payload["id_slot"] == 0
    assert metrics["details"]["slot_router"]["last_decision"]["reason"] == "sticky_existing"
    assert metrics["details"]["slot_router"]["bindings"][0]["slot_id"] == 0


def test_llama_cpp_adapter_leaves_short_or_unkeyed_requests_unassigned(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32105",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=2,
        sticky_session_prompt_threshold=1000,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []

    def fake_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={"object": "chat.completion", "choices": [{"message": {"content": "OK"}}]},
                headers={"content-type": "application/json"},
            )
        if path == "slots":
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "props":
            return FakeResponse(
                json_data={"default_generation_settings": {"n_ctx": 16384}},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={"model": "qwen35-4b-gguf", "messages": [{"role": "user", "content": "ping"}]},
    )

    request_payload = [call for call in calls if call[1] == "v1/chat/completions"][0][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert "id_slot" not in request_payload
    assert request_payload["cache_prompt"] is True
    assert metrics["details"]["slot_router"]["last_decision"]["reason"] in {
        "no_routing_key",
        "short_prompt",
    }


def test_llama_cpp_adapter_steers_short_request_away_from_sticky_slot(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32106",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=2,
        sticky_session_prompt_threshold=8,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []

    def fake_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 16384}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={"object": "chat.completion", "choices": [{"message": {"content": "OK"}}]},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix " * 64}],
            "metadata": {"conversation_id": "long-session"},
        },
    )
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "short-session"},
        },
    )

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    long_payload = request_calls[0][2]["json"]
    short_payload = request_calls[1][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert long_payload["id_slot"] == 0
    assert short_payload["id_slot"] == 1
    assert (
        metrics["details"]["slot_router"]["last_decision"]["reason"]
        == "short_prompt_unowned_slot"
    )


def test_llama_cpp_adapter_recycles_stale_owned_slot_for_unkeyed_short_request(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32107",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=2,
        sticky_session_prompt_threshold=8,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []

    def fake_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 16384}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={"object": "chat.completion", "choices": [{"message": {"content": "OK"}}]},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix " * 64}],
            "metadata": {"conversation_id": "long-session"},
        },
    )
    with adapter._lock:
        adapter._bind_session_locked(
            "stale-session",
            slot_id=1,
            estimated_prompt_tokens=4096,
        )
        adapter._session_bindings["stale-session"].last_used_at -= 10

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={"model": "qwen35-4b-gguf", "messages": [{"role": "user", "content": "ping"}]},
    )

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    short_payload = request_calls[-1][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert short_payload["id_slot"] == 1
    assert (
        metrics["details"]["slot_router"]["last_decision"]["reason"]
        == "unkeyed_recycled_slot"
    )
    assert metrics["details"]["slot_router"]["bindings"][0]["slot_id"] == 0


def test_llama_cpp_adapter_recycles_stale_owned_slot_for_keyed_short_request(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32108",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=2,
        sticky_session_prompt_threshold=8,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []

    def fake_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 16384}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={"object": "chat.completion", "choices": [{"message": {"content": "OK"}}]},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix " * 64}],
            "metadata": {"conversation_id": "long-session"},
        },
    )
    with adapter._lock:
        adapter._bind_session_locked(
            "stale-session",
            slot_id=1,
            estimated_prompt_tokens=4096,
        )
        adapter._session_bindings["stale-session"].last_used_at -= 10

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "short-session"},
        },
    )

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    short_payload = request_calls[-1][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert short_payload["id_slot"] == 1
    assert (
        metrics["details"]["slot_router"]["last_decision"]["reason"]
        == "short_prompt_recycled_slot"
    )
    assert metrics["details"]["slot_router"]["bindings"][0]["slot_id"] == 0
