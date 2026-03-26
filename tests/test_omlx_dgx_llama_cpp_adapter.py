# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import omlx_dgx.runtime.backend as backend_module
import omlx_dgx.runtime.llama_cpp as llama_cpp_module
from omlx_dgx.config import (
    BackendConfig,
    DGXRuntimeConfig,
    LlamaCppModelPoolConfig,
    LlamaCppModelRegistration,
    ModelProfile,
)
from omlx_dgx.runtime.llama_cpp import LlamaCppBackendAdapter, LlamaCppModelPoolAdapter


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
    assert "--cache-reuse" not in command
    assert "--checkpoint-every-n-tokens" in command
    assert command[command.index("--checkpoint-every-n-tokens") + 1] == "1024"
    assert "--ctx-checkpoints" in command
    assert command[command.index("--ctx-checkpoints") + 1] == "64"
    assert "--slot-prompt-similarity" in command
    assert command[command.index("--slot-prompt-similarity") + 1] == "0.25"
    assert "--metrics" in command
    assert "--slot-save-path" in command
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
    assert diagnostics["configured_cache_reuse"] == 256
    assert diagnostics["cache_reuse"] == 0
    assert diagnostics["cache_reuse_supported"] is False
    assert "Qwen3.5 uses the hybrid GDN/Mamba-recurrent cache path" in diagnostics["cache_reuse_blocker"]
    assert diagnostics["enable_session_stickiness"] is True
    assert diagnostics["enable_session_restore"] is True
    assert diagnostics["session_restore_min_prompt_tokens"] == 2048
    assert diagnostics["sticky_session_prompt_threshold"] == 2048
    assert diagnostics["single_session_continuation_enabled"] is False
    assert diagnostics["single_session_continuation_ttl_seconds"] == 600
    assert diagnostics["slot_save_path"].endswith("/runtime/slot_saves")


def test_llama_cpp_adapter_keeps_cache_reuse_for_non_qwen35_models(tmp_path: Path):
    gguf_path = tmp_path / "TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32120",
        quant_mode="gguf_experimental",
        model_source="gguf",
        artifact_path=str(gguf_path),
        cache_reuse=256,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert "--cache-reuse" in command
    assert command[command.index("--cache-reuse") + 1] == "256"
    assert diagnostics["configured_cache_reuse"] == 256
    assert diagnostics["cache_reuse"] == 256
    assert diagnostics["cache_reuse_supported"] is True
    assert diagnostics["cache_reuse_blocker"] == ""


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


def test_llama_cpp_rerank_model_enables_reranking_endpoint(tmp_path: Path):
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32119",
        quant_mode="gguf_experimental",
        model_source="gguf",
        model_repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    capabilities = adapter.capabilities()

    assert "--reranking" in command
    assert capabilities.rerank is True
    assert capabilities.chat_completions is False
    assert capabilities.completions is False


def test_llama_cpp_mmproj_enables_multimodal_launch_and_capabilities(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-35B-A3B.Q4_K_M.gguf"
    mmproj_path = tmp_path / "Qwen3.5-35B-A3B.mmproj-Q8_0.gguf"
    gguf_path.write_bytes(b"GGUF")
    mmproj_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32121",
        quant_mode="gguf_experimental",
        model_source="gguf",
        artifact_path=str(gguf_path),
        mmproj_path=str(mmproj_path),
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    capabilities = adapter.capabilities()

    assert "--mmproj" in command
    assert str(mmproj_path) in command
    assert capabilities.vision_chat is True
    assert capabilities.ocr is True


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
    assert metrics["details"]["continuation"]["enabled"] is True
    assert metrics["details"]["session_restore"]["enabled"] is True
    assert metrics["details"]["metrics_excerpt"][0] == "# HELP llama_tokens tokens"
    assert metrics["details"]["telemetry"]["gpu_metrics_source"] == "nvidia-smi"
    assert cache_report["props"]["model_path"] == str(gguf_path)
    assert cache_report["slots"]["slots"][0]["id"] == 0
    assert cache_report["slot_router"]["slot_summary"][0]["id"] == 0
    assert cache_report["continuation"]["ttl_seconds"] == 600
    assert cache_report["session_restore"]["slot_save_path"].endswith("/runtime/slot_saves")


def test_llama_cpp_model_pool_external_openai_embedding_registration(tmp_path: Path):
    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32120",
            model_pool=LlamaCppModelPoolConfig(
                models={
                    "nomic-embed": LlamaCppModelRegistration(
                        model_id="nomic-embed",
                        model_alias="embed-text",
                        backend_kind="openai_compatible",
                        backend_model_name="text-embedding-nomic-embed-text-v1.5",
                        base_url="http://127.0.0.1:1234",
                        supports_embeddings=True,
                        pinned=True,
                    )
                }
            ),
        ),
        models={
            "nomic-embed": ModelProfile(
                model_id="nomic-embed",
                model_alias="embed-text",
                is_default=True,
                supports_embeddings=True,
            )
        },
    )
    adapter = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)
    handle = adapter._models["nomic-embed"]

    def fake_request(method, path, timeout=0, **kwargs):
        assert path == "v1/embeddings"
        assert kwargs["json"]["model"] == "text-embedding-nomic-embed-text-v1.5"
        return FakeResponse(
            json_data={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": kwargs["json"]["model"],
            },
            headers={"content-type": "application/json"},
        )

    handle.adapter._request = fake_request  # type: ignore[attr-defined,method-assign]

    response = adapter.proxy(
        "POST",
        "v1/embeddings",
        json={"model": "embed-text", "input": "hello"},
    )

    assert response.json()["model"] == "text-embedding-nomic-embed-text-v1.5"
    assert adapter.capabilities().embeddings is True
    assert adapter.capabilities().chat_completions is False


def test_llama_cpp_model_pool_remote_repo_registration_does_not_inherit_parent_artifact(
    tmp_path: Path,
):
    parent_model = tmp_path / "Qwen3.5-35B-A3B.Q4_K_M.gguf"
    parent_model.write_bytes(b"GGUF")
    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32122",
            quant_mode="gguf_experimental",
            model_source="gguf",
            artifact_path=str(parent_model),
            model_pool=LlamaCppModelPoolConfig(
                models={
                    "rerank": LlamaCppModelRegistration(
                        model_id="rerank",
                        model_alias="rerank-qwen",
                        model_repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
                        base_url="http://127.0.0.1:32123",
                        supports_rerank=True,
                        pinned=True,
                    )
                }
            ),
        ),
        models={
            "qwen35-35b": ModelProfile(
                model_id="qwen35-35b",
                model_alias="qwen35-35b",
                is_default=True,
            ),
            "rerank": ModelProfile(
                model_id="rerank",
                model_alias="rerank-qwen",
                supports_rerank=True,
            )
        },
    )

    adapter = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)
    handle = adapter._models["rerank"]

    assert handle.registration.artifact_path == ""
    assert handle.registration.model_repo_id == "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"
    assert handle.adapter.config.artifact_path == ""
    assert handle.adapter.config.model_repo_id == "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF"


def test_llama_cpp_model_pool_secondary_registration_does_not_inherit_parent_mmproj(
    tmp_path: Path,
):
    parent_model = tmp_path / "Qwen3.5-35B-A3B.Q4_K_M.gguf"
    parent_model.write_bytes(b"GGUF")
    parent_mmproj = tmp_path / "Qwen3.5-35B-A3B.mmproj-Q8_0.gguf"
    parent_mmproj.write_bytes(b"MM")
    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32122",
            quant_mode="gguf_experimental",
            model_source="gguf",
            artifact_path=str(parent_model),
            mmproj_path=str(parent_mmproj),
            model_pool=LlamaCppModelPoolConfig(
                models={
                    "rerank": LlamaCppModelRegistration(
                        model_id="rerank",
                        model_alias="rerank-qwen",
                        model_repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
                        base_url="http://127.0.0.1:32123",
                        supports_rerank=True,
                        pinned=True,
                    )
                }
            ),
        ),
        models={
            "qwen35-35b": ModelProfile(
                model_id="qwen35-35b",
                model_alias="qwen35-35b",
                is_default=True,
            ),
            "rerank": ModelProfile(
                model_id="rerank",
                model_alias="rerank-qwen",
                supports_rerank=True,
            )
        },
    )

    adapter = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)
    handle = adapter._models["rerank"]

    assert handle.registration.mmproj_path == ""
    assert handle.adapter.config.mmproj_path == ""


def test_llama_cpp_model_pool_primary_registration_inherits_profile_capabilities(
    tmp_path: Path,
):
    parent_model = tmp_path / "Qwen3.5-35B-A3B.Q4_K_M.gguf"
    parent_model.write_bytes(b"GGUF")
    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32122",
            quant_mode="gguf_experimental",
            model_source="gguf",
            artifact_path=str(parent_model),
        ),
        models={
            "qwen35-35b": ModelProfile(
                model_id="qwen35-35b",
                model_alias="qwen35-35b",
                is_default=True,
                supports_vision=True,
                primary_service="chat",
            )
        },
    )

    adapter = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)
    handle = adapter._models["qwen35-35b"]

    assert handle.registration.supports_vision is True
    assert handle.registration.primary_service == "chat"


def test_llama_cpp_adapter_surfaces_nvml_errors_in_telemetry(tmp_path: Path, monkeypatch):
    gguf_path = tmp_path / "Qwen3.5-4B-Q6_K.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32118",
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
                json_data={"default_generation_settings": {"n_ctx": 32768}},
                headers={"content-type": "application/json"},
            )
        if path == "slots":
            return FakeResponse(
                json_data={"slots": [{"id": 0, "state": "idle"}]},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["nvidia-smi"],
            stderr="Failed to initialize NVML: Unknown Error",
        )

    adapter._request = fake_request  # type: ignore[method-assign]
    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)

    metrics = adapter.collect_metrics().to_dict()
    telemetry = metrics["details"]["telemetry"]

    assert telemetry["gpu_metrics_ok"] is False
    assert "Unknown Error" in telemetry["gpu_metrics_error"]
    assert "MemTotal" in telemetry["system_memory_kb"]


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


def test_llama_cpp_adapter_single_slot_continuation_hits_followup(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32109",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    first_payload = {
        "model": "qwen35-4b-gguf",
        "messages": [{"role": "user", "content": "prefix " * 64}],
        "metadata": {"conversation_id": "session-a"},
    }
    second_payload = {
        "model": "qwen35-4b-gguf",
        "messages": [
            {"role": "user", "content": "prefix " * 64},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "benefit?"},
        ],
        "metadata": {"conversation_id": "session-a"},
    }

    adapter.proxy("POST", "v1/chat/completions", json=first_payload)
    adapter.proxy("POST", "v1/chat/completions", json=second_payload)

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    assert request_calls[0][2]["json"]["id_slot"] == 0
    assert request_calls[1][2]["json"]["id_slot"] == 0
    assert request_calls[1][2]["json"]["messages"] == [
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "benefit?"},
    ]

    metrics = adapter.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]

    assert continuation["last_decision"]["reason"] == "hit"
    assert continuation["last_decision"]["continuation_hit"] is True
    assert continuation["last_decision"]["suffix_only"] is True
    assert continuation["state"]["slot_id"] == 0
    assert continuation["state"]["message_count"] == 3
    assert continuation["state"]["slot_message_count"] == 4


def test_llama_cpp_adapter_single_slot_prefix_drift_forces_cold_restart(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32110",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix A " * 64}],
            "metadata": {"conversation_id": "session-a"},
        },
    )
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix B " * 64}],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    assert request_calls[1][2]["json"]["id_slot"] == 0

    metrics = adapter.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]

    assert continuation["last_decision"]["reason"] == "prefix_drift"
    assert continuation["last_decision"]["prefix_drift"] is True
    assert continuation["last_decision"]["continuation_hit"] is False


def test_llama_cpp_adapter_single_slot_without_conversation_id_passes_through(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32111",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
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
    continuation = metrics["details"]["continuation"]

    assert "id_slot" not in request_payload
    assert continuation["last_decision"]["reason"] == "no_conversation_id"


def test_llama_cpp_adapter_single_slot_repeat_prompt_keeps_full_messages(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32112",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
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
    second_payload = request_calls[1][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert second_payload["messages"] == payload["messages"]
    assert metrics["details"]["continuation"]["last_decision"]["suffix_only"] is False


def test_llama_cpp_adapter_restores_exact_cold_cache_for_new_conversation(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32116",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        session_restore_min_prompt_tokens=0,
        sticky_session_prompt_threshold=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_saved": 128,
                    "n_written": 4096,
                    "timings": {"save_ms": 5.0},
                },
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=restore":
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_restored": 128,
                    "n_read": 4096,
                    "timings": {"restore_ms": 4.0},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "turn1"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    payload = {
        "model": "qwen35-4b-gguf",
        "messages": [{"role": "user", "content": "prefix " * 128}],
        "metadata": {"conversation_id": "session-a"},
    }
    adapter.proxy("POST", "v1/chat/completions", json=payload)

    with adapter._lock:
        adapter._single_slot_continuation = None
        adapter._single_slot_continuation_dirty = False

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            **payload,
            "metadata": {"conversation_id": "session-b"},
        },
    )

    metrics = adapter.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]
    cold_cache = metrics["details"]["cold_cache"]
    request_payloads = [call[2]["json"] for call in calls if call[1] == "v1/chat/completions"]

    assert any(call[1] == "slots/0?action=restore" for call in calls)
    assert request_payloads[1]["messages"] == payload["messages"]
    assert continuation["last_decision"]["reason"] == "cold_cache_hit"
    assert continuation["last_decision"]["continuation_hit"] is True
    assert cold_cache["counts"]["exact_hits"] == 1
    assert cold_cache["last_hit"]["source"] == "exact"
    assert cold_cache["last_hit"]["ok"] is True


def test_llama_cpp_adapter_restores_shared_prefix_for_new_conversation(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32117",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        session_restore_min_prompt_tokens=0,
        sticky_session_prompt_threshold=1,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []
    replies = iter(["turn1", "turn2"])

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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_saved": 160,
                    "n_written": 6144,
                    "timings": {"save_ms": 5.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=restore":
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_restored": 160,
                    "n_read": 6144,
                    "timings": {"restore_ms": 4.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [
                        {"message": {"role": "assistant", "content": next(replies)}}
                    ],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix " * 128}],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    with adapter._lock:
        adapter._single_slot_continuation = None
        adapter._single_slot_continuation_dirty = False

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix " * 128},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-b"},
        },
    )

    metrics = adapter.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]
    cold_cache = metrics["details"]["cold_cache"]
    request_payloads = [call[2]["json"] for call in calls if call[1] == "v1/chat/completions"]

    assert any(call[1] == "slots/0?action=restore" for call in calls)
    assert request_payloads[1]["messages"] == [
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "benefit?"},
    ]
    assert continuation["last_decision"]["reason"] == "shared_prefix_hit"
    assert continuation["last_decision"]["continuation_hit"] is True
    assert continuation["last_decision"]["suffix_only"] is True
    assert cold_cache["counts"]["prefix_hits"] == 1
    assert cold_cache["last_hit"]["source"] == "prefix"
    assert cold_cache["last_hit"]["ok"] is True


def test_llama_cpp_adapter_saves_and_restores_named_context(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32118",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        session_restore_min_prompt_tokens=0,
        sticky_session_prompt_threshold=1,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []
    replies = iter(["turn1", "turn2"])

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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_saved": 192,
                    "n_written": 7168,
                    "timings": {"save_ms": 6.0},
                },
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=restore":
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_restored": 192,
                    "n_read": 7168,
                    "timings": {"restore_ms": 4.25},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [
                        {"message": {"role": "assistant", "content": next(replies)}}
                    ],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]
    prefix = {"role": "user", "content": "prefix " * 128}
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [prefix],
            "metadata": {"context_id": "ctx-a"},
        },
    )

    named_contexts = adapter.list_named_contexts()
    assert named_contexts["stats"]["snapshots"] == 1
    assert named_contexts["contexts"][0]["context_id"] == "ctx-a"

    with adapter._lock:
        adapter._single_slot_continuation = None
        adapter._single_slot_continuation_dirty = False

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                prefix,
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"context_id": "ctx-a"},
        },
    )

    request_payloads = [call[2]["json"] for call in calls if call[1] == "v1/chat/completions"]
    metrics = adapter.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]
    named_context_summary = metrics["details"]["named_contexts"]

    assert any(call[1] == "slots/0?action=restore" for call in calls)
    assert request_payloads[1]["messages"] == [
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "benefit?"},
    ]
    assert continuation["last_decision"]["reason"] == "named_context_hit"
    assert continuation["last_decision"]["continuation_hit"] is True
    assert continuation["last_decision"]["suffix_only"] is True
    assert named_context_summary["counts"]["saved"] >= 1
    assert named_context_summary["counts"]["restored"] == 1
    assert named_context_summary["last_restore"]["ok"] is True


def test_llama_cpp_adapter_deletes_named_context_slot_save(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32118",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        session_restore_min_prompt_tokens=0,
        sticky_session_prompt_threshold=1,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)

    def fake_request(method, path, timeout=0, **kwargs):
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_saved": 96,
                    "n_written": 3072,
                    "timings": {"save_ms": 3.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "turn1"}}],
                },
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
            "metadata": {"context_id": "ctx-a"},
        },
    )

    named_context = adapter.get_named_context("ctx-a")
    slot_save_file = adapter._slot_save_path / named_context["save_filename"]
    assert slot_save_file.exists()

    deleted = adapter.delete_named_context("ctx-a")
    named_contexts = adapter.list_named_contexts()

    assert deleted["deleted"] is True
    assert not slot_save_file.exists()
    assert named_contexts["stats"]["snapshots"] == 0
    assert named_contexts["counts"]["deleted"] == 1


def test_llama_cpp_adapter_single_slot_followup_keeps_last_turn_anchor(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32117",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []
    replies = iter(["turn1", "turn2", "turn3"])

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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": next(replies),
                            }
                        }
                    ],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "prefix"}],
            "metadata": {"conversation_id": "session-a"},
        },
    )
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix"},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix"},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
                {"role": "assistant", "content": "turn2"},
                {"role": "user", "content": "again"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    third_payload = request_calls[2][2]["json"]
    metrics = adapter.collect_metrics().to_dict()

    assert third_payload["messages"] == [
        {"role": "assistant", "content": "turn2"},
        {"role": "user", "content": "again"},
    ]
    assert metrics["details"]["continuation"]["last_decision"]["suffix_only"] is True


def test_llama_cpp_adapter_normalizes_disable_thinking_payload(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32115",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "session-a"},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )

    request_payload = [call for call in calls if call[1] == "v1/chat/completions"][0][2]["json"]

    assert request_payload["chat_template_kwargs"]["enable_thinking"] is False
    assert request_payload["enableThinking"] is False
    assert request_payload["reasoning"] is False
    assert request_payload["reasoning_budget"] == 0
    assert request_payload["reasoning_format"] == "none"
    assert request_payload["thinking_forced_open"] is False


def test_llama_cpp_adapter_preserves_explicit_reasoning_override(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32116",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "session-a"},
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "deepseek",
            "reasoning_budget": -1,
        },
    )

    request_payload = [call for call in calls if call[1] == "v1/chat/completions"][0][2]["json"]

    assert request_payload["reasoning_format"] == "deepseek"
    assert request_payload["reasoning_budget"] == -1
    assert request_payload["enableThinking"] is False
    assert request_payload["reasoning"] is False


def test_llama_cpp_adapter_normalizes_disable_thinking_from_extra_body(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32122",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
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
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "extra_body": {"think": False},
        },
    )

    request_payload = [call for call in calls if call[1] == "v1/chat/completions"][0][2]["json"]

    assert "extra_body" not in request_payload
    assert "think" not in request_payload
    assert request_payload["chat_template_kwargs"]["enable_thinking"] is False
    assert request_payload["enableThinking"] is False
    assert request_payload["reasoning"] is False
    assert request_payload["reasoning_format"] == "none"


def test_llama_cpp_adapter_strips_disabled_thinking_tags_and_preserves_continuation(
    tmp_path: Path,
):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32121",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        enable_session_restore=False,
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
            response_text = "<think>\n\n</think>\n\nAnswer"
            if len([call for call in calls if call[1] == "v1/chat/completions"]) > 1:
                response_text = "<think>\n\n</think>\n\nBenefit"
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": response_text}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]

    first_response = adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "session-a"},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert first_response.json()["choices"][0]["message"]["content"] == "Answer"

    second_response = adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "ping"},
                {"role": "assistant", "content": "Answer"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-a"},
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    assert second_response.json()["choices"][0]["message"]["content"] == "Benefit"

    request_calls = [call for call in calls if call[1] == "v1/chat/completions"]
    assert request_calls[1][2]["json"]["messages"] == [
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "benefit?"},
    ]

    metrics = adapter.collect_metrics().to_dict()
    assert metrics["details"]["continuation"]["last_decision"]["reason"] == "hit"
    assert metrics["details"]["continuation"]["last_decision"]["continuation_hit"] is True


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


def test_llama_cpp_adapter_restores_single_slot_continuation_after_adapter_restart(
    tmp_path: Path,
):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32119",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    adapter.process_manager.pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def fake_request(method, path, timeout=0, **kwargs):
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 65536}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "session-a.bin",
                    "n_saved": 128,
                    "n_written": 4096,
                    "timings": {"save_ms": 5.25},
                },
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=restore":
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "session-a.bin",
                    "n_restored": 128,
                    "n_read": 4096,
                    "timings": {"restore_ms": 4.0},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
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
            "metadata": {"conversation_id": "session-a"},
        },
    )

    restored = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    restored.process_manager.pid_path.write_text(str(os.getpid()), encoding="utf-8")
    summary_before = restored._single_slot_summary()
    calls = []

    def fake_restored_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        return fake_request(method, path, timeout=timeout, **kwargs)

    restored._request = fake_restored_request  # type: ignore[method-assign]
    restored.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix " * 64},
                {"role": "assistant", "content": "OK"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    request_payload = [call for call in calls if call[1] == "v1/chat/completions"][0][2]["json"]
    metrics = restored.collect_metrics().to_dict()
    session_restore = metrics["details"]["session_restore"]

    assert summary_before["restored_from_disk"] is True
    assert summary_before["recovery_pending"] is False
    assert not any(call[1] == "slots/0?action=save" for call in calls)
    assert request_payload["messages"] == [
        {"role": "assistant", "content": "OK"},
        {"role": "user", "content": "benefit?"},
    ]
    assert metrics["details"]["continuation"]["last_decision"]["reason"] == "hit"
    assert metrics["details"]["continuation"]["durable_state_dirty"] is True
    assert session_restore["stats"]["snapshots"] == 1
    assert session_restore["last_save"] is None
    assert session_restore["last_restore"] is None


def test_llama_cpp_adapter_restores_saved_slot_after_runtime_restart(
    tmp_path: Path,
    monkeypatch,
):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32120",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
    )
    monkeypatch.setattr(llama_cpp_module.os, "kill", lambda pid, sig: None)

    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    adapter.process_manager.pid_path.write_text("1111", encoding="utf-8")

    def fake_request(method, path, timeout=0, **kwargs):
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 65536}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "session-a.bin",
                    "n_saved": 256,
                    "n_written": 8192,
                    "timings": {"save_ms": 6.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=restore":
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "session-a.bin",
                    "n_restored": 256,
                    "n_read": 8192,
                    "timings": {"restore_ms": 5.75},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "turn1"}}],
                },
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
            "metadata": {"conversation_id": "session-a"},
        },
    )

    adapter.process_manager.pid_path.write_text("2222", encoding="utf-8")
    restored = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    restored.process_manager.pid_path.write_text("2222", encoding="utf-8")
    summary_before = restored._single_slot_summary()
    calls = []
    replies = iter(["turn2", "turn3"])

    def fake_restored_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": next(replies),
                            }
                        }
                    ],
                },
                headers={"content-type": "application/json"},
            )
        return fake_request(method, path, timeout=timeout, **kwargs)

    restored._request = fake_restored_request  # type: ignore[method-assign]
    restored.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix " * 64},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )
    restored.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix " * 64},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
                {"role": "assistant", "content": "turn2"},
                {"role": "user", "content": "again"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    request_payloads = [call[2]["json"] for call in calls if call[1] == "v1/chat/completions"]
    metrics = restored.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]
    session_restore = metrics["details"]["session_restore"]

    assert summary_before["restored_from_disk"] is True
    assert summary_before["recovery_pending"] is True
    assert any(call[1] == "slots/0?action=restore" for call in calls)
    assert request_payloads[0]["messages"] == [
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "benefit?"},
    ]
    assert request_payloads[1]["messages"] == [
        {"role": "assistant", "content": "turn2"},
        {"role": "user", "content": "again"},
    ]
    assert continuation["counts"]["recovery_replay"] == 0
    assert continuation["last_decision"]["reason"] == "hit"
    assert session_restore["last_restore"]["ok"] is True
    assert session_restore["last_restore"]["status"] == "restored"
    assert session_restore["counts"]["restored"] == 1


def test_llama_cpp_adapter_falls_back_to_cold_replay_when_slot_restore_fails(
    tmp_path: Path,
    monkeypatch,
):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32120",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
    )
    monkeypatch.setattr(llama_cpp_module.os, "kill", lambda pid, sig: None)

    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    adapter.process_manager.pid_path.write_text("1111", encoding="utf-8")

    def fake_request(method, path, timeout=0, **kwargs):
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 65536}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "session-a.bin",
                    "n_saved": 256,
                    "n_written": 8192,
                    "timings": {"save_ms": 6.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "turn1"}}],
                },
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
            "metadata": {"conversation_id": "session-a"},
        },
    )

    adapter.process_manager.pid_path.write_text("2222", encoding="utf-8")
    restored = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    restored.process_manager.pid_path.write_text("2222", encoding="utf-8")
    calls = []

    def fake_restored_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path == "slots/0?action=restore":
            return FakeResponse(status_code=400, text="restore failed")
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "turn2"}}],
                },
                headers={"content-type": "application/json"},
            )
        return fake_request(method, path, timeout=timeout, **kwargs)

    restored._request = fake_restored_request  # type: ignore[method-assign]
    restored.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": "prefix " * 64},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    request_payload = [call[2]["json"] for call in calls if call[1] == "v1/chat/completions"][0]
    metrics = restored.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]
    session_restore = metrics["details"]["session_restore"]

    assert any(call[1] == "slots/0?action=restore" for call in calls)
    assert request_payload["messages"] == [
        {"role": "user", "content": "prefix " * 64},
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "benefit?"},
    ]
    assert continuation["counts"]["recovery_replay"] == 1
    assert continuation["last_decision"]["reason"] == "recovery_replay"
    assert session_restore["last_restore"]["ok"] is False
    assert session_restore["counts"]["restore_fallback"] == 1


def test_llama_cpp_adapter_disables_slot_restore_when_backend_reports_unsupported(
    tmp_path: Path,
):
    gguf_path = tmp_path / "Qwen3.5-35B-A3B.Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32120",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        session_restore_min_prompt_tokens=0,
        sticky_session_prompt_threshold=1,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    calls = []
    replies = iter(["turn1", "turn2"])

    def fake_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-35b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 65536}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            return FakeResponse(
                status_code=501,
                text='{"error":{"code":501,"message":"This feature is not supported by multimodal","type":"not_supported_error"}}',
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [
                        {"message": {"role": "assistant", "content": next(replies)}}
                    ],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-35b",
            "messages": [{"role": "user", "content": "prefix " * 64}],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    with adapter._lock:
        adapter._single_slot_recovery_pending = True

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-35b",
            "messages": [
                {"role": "user", "content": "prefix " * 64},
                {"role": "assistant", "content": "turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    request_payloads = [call[2]["json"] for call in calls if call[1] == "v1/chat/completions"]
    metrics = adapter.collect_metrics().to_dict()
    continuation = metrics["details"]["continuation"]
    session_restore = metrics["details"]["session_restore"]

    assert not any(call[1] == "slots/0?action=restore" for call in calls)
    assert request_payloads[1]["messages"] == [
        {"role": "user", "content": "prefix " * 64},
        {"role": "assistant", "content": "turn1"},
        {"role": "user", "content": "benefit?"},
    ]
    assert session_restore["enabled"] is False
    assert session_restore["supported"] is False
    assert "multimodal" in session_restore["disabled_reason"]
    assert session_restore["counts"]["unsupported"] == 1
    assert session_restore["last_save"]["status"] == "unsupported"
    assert session_restore["last_restore"] is None
    assert continuation["last_decision"]["reason"] == "recovery_replay"
    assert continuation["counts"]["recovery_replay"] == 1


def test_llama_cpp_adapter_defers_short_followup_slot_save_until_stop(tmp_path: Path):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32120",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=1,
        sticky_session_prompt_threshold=2048,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    save_calls = []
    long_prefix = "prefix " * 4096

    def fake_request(method, path, timeout=0, **kwargs):
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path == "slots/0?action=save":
            save_calls.append(kwargs["json"]["filename"])
            (adapter._slot_save_path / kwargs["json"]["filename"]).write_bytes(b"slot-save")
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": kwargs["json"]["filename"],
                    "n_saved": 42,
                    "n_written": 1234,
                    "timings": {"save_ms": 12.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "1"}}],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    adapter._request = fake_request  # type: ignore[method-assign]
    adapter.process_manager.stop = lambda: {"ok": True, "reason": "stopped"}  # type: ignore[method-assign]

    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": long_prefix}],
            "metadata": {"conversation_id": "session-a"},
        },
    )
    adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [
                {"role": "user", "content": long_prefix},
                {"role": "assistant", "content": "1"},
                {"role": "user", "content": "2"},
            ],
            "metadata": {"conversation_id": "session-a"},
        },
    )

    continuation_before = adapter.collect_metrics().to_dict()["details"]["continuation"]
    assert len(save_calls) == 1
    assert continuation_before["durable_state_dirty"] is True

    adapter.stop_runtime()

    continuation_after = adapter.collect_metrics().to_dict()["details"]["continuation"]
    assert len(save_calls) == 2
    assert continuation_after["durable_state_dirty"] is False


def test_llama_cpp_adapter_restores_sticky_bindings_after_adapter_restart(
    tmp_path: Path,
):
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="llama_cpp",
        base_url="http://127.0.0.1:32121",
        artifact_path=str(gguf_path),
        quant_mode="gguf_experimental",
        model_source="gguf",
        parallel_slots=2,
        sticky_session_prompt_threshold=8,
    )
    adapter = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    adapter.process_manager.pid_path.write_text(str(os.getpid()), encoding="utf-8")

    def fake_request(method, path, timeout=0, **kwargs):
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "qwen3.5-4b"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"slots", "props"}:
            if path == "props":
                return FakeResponse(
                    json_data={"default_generation_settings": {"n_ctx": 65536}},
                    headers={"content-type": "application/json"},
                )
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}, {"id": 1, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path.startswith("slots/") and "?action=save" in path:
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "sticky.bin",
                    "n_saved": 128,
                    "n_written": 4096,
                    "timings": {"save_ms": 4.5},
                },
                headers={"content-type": "application/json"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                },
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

    restored = LlamaCppBackendAdapter.from_backend_config(config, tmp_path)
    restored.process_manager.pid_path.write_text(str(os.getpid()), encoding="utf-8")
    calls = []

    def fake_restored_request(method, path, timeout=0, **kwargs):
        calls.append((method, path, kwargs))
        return fake_request(method, path, timeout=timeout, **kwargs)

    restored._request = fake_restored_request  # type: ignore[method-assign]
    restored.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35-4b-gguf",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "short-session"},
        },
    )

    request_payload = [call for call in calls if call[1] == "v1/chat/completions"][0][2]["json"]
    metrics = restored.collect_metrics().to_dict()

    assert request_payload["id_slot"] == 1
    assert (
        metrics["details"]["slot_router"]["last_decision"]["reason"]
        == "short_prompt_unowned_slot"
    )


def _install_pool_adapter_fakes(
    pool: LlamaCppModelPoolAdapter,
    model_id: str,
    request_handler,
) -> None:
    handle = pool._models[model_id]
    handle.adapter.start_runtime = lambda: {"started": True, "model_id": model_id}  # type: ignore[method-assign]
    handle.adapter.stop_runtime = lambda: {"stopped": True, "model_id": model_id}  # type: ignore[method-assign]
    handle.adapter._request = request_handler  # type: ignore[method-assign]


def _fake_pool_request(model_name: str, replies):
    reply_iter = iter(replies)

    def _request(method, path, timeout=0, **kwargs):
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": model_name}]},
                headers={"content-type": "application/json"},
            )
        if path == "props":
            return FakeResponse(
                json_data={"default_generation_settings": {"n_ctx": 65536}},
                headers={"content-type": "application/json"},
            )
        if path == "slots":
            return FakeResponse(
                json_data=[{"id": 0, "is_processing": False}],
                headers={"content-type": "application/json"},
            )
        if path.startswith("slots/") and "?action=save" in path:
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "pool.bin",
                    "n_saved": 96,
                    "n_written": 3072,
                    "timings": {"save_ms": 3.25},
                },
                headers={"content-type": "application/json"},
            )
        if path.startswith("slots/") and "?action=restore" in path:
            return FakeResponse(
                json_data={
                    "id_slot": 0,
                    "filename": "pool.bin",
                    "n_restored": 96,
                    "n_read": 3072,
                    "timings": {"restore_ms": 2.75},
                },
                headers={"content-type": "application/json"},
            )
        if path == "metrics":
            return FakeResponse(
                text="# HELP llama_tokens tokens\nllama_tokens 42\n",
                headers={"content-type": "text/plain"},
            )
        if path == "v1/chat/completions":
            return FakeResponse(
                json_data={
                    "object": "chat.completion",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": next(reply_iter),
                            }
                        }
                    ],
                },
                headers={"content-type": "application/json"},
            )
        raise AssertionError(path)

    return _request


def test_llama_cpp_model_pool_preserves_primary_continuation_when_secondary_loads(tmp_path: Path):
    primary_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    secondary_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    primary_path.write_bytes(b"GGUF")
    secondary_path.write_bytes(b"GGUF")

    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32200",
            quant_mode="gguf_experimental",
            model_source="gguf",
            artifact_path=str(primary_path),
            model_repo_id="primary",
            model_pool=LlamaCppModelPoolConfig(
                max_loaded_models=2,
                models={
                    "secondary": LlamaCppModelRegistration(
                        model_id="secondary",
                        artifact_path=str(secondary_path),
                        base_url="http://127.0.0.1:32201",
                        gguf_variant="Q4_K_S",
                    )
                },
            ),
        ),
        models={
            "primary": ModelProfile(model_id="primary", is_default=True),
            "secondary": ModelProfile(model_id="secondary"),
        },
    )
    pool = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)

    _install_pool_adapter_fakes(
        pool,
        "primary",
        _fake_pool_request("primary", ["primary-turn1", "primary-turn2"]),
    )
    _install_pool_adapter_fakes(
        pool,
        "secondary",
        _fake_pool_request("secondary", ["secondary-turn1"]),
    )

    pool.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "primary",
            "messages": [{"role": "user", "content": "prefix " * 64}],
            "metadata": {"conversation_id": "primary-session"},
        },
    )
    pool.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "secondary",
            "messages": [{"role": "user", "content": "ping"}],
            "metadata": {"conversation_id": "secondary-session"},
        },
    )
    pool.unload_model("secondary", reason="manual_unload")
    pool.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "primary",
            "messages": [
                {"role": "user", "content": "prefix " * 64},
                {"role": "assistant", "content": "primary-turn1"},
                {"role": "user", "content": "benefit?"},
            ],
            "metadata": {"conversation_id": "primary-session"},
        },
    )

    continuation = pool._models["primary"].adapter._single_slot_summary()
    diagnostics = pool.model_pool_diagnostics()
    primary_state = next(
        model for model in diagnostics["models"] if model["model_id"] == "primary"
    )
    secondary_state = next(
        model for model in diagnostics["models"] if model["model_id"] == "secondary"
    )

    assert continuation["last_decision"]["reason"] == "hit"
    assert continuation["last_decision"]["continuation_hit"] is True
    assert pool._models["primary"].loaded is True
    assert primary_state["model_repo_id"] == ""
    assert secondary_state["loaded"] is False
    assert secondary_state["last_unload_reason"] == "manual_unload"


def test_llama_cpp_model_pool_evicts_lru_model_when_loading_new_one(tmp_path: Path):
    primary_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    secondary_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    primary_path.write_bytes(b"GGUF")
    secondary_path.write_bytes(b"GGUF")

    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32210",
            quant_mode="gguf_experimental",
            model_source="gguf",
            artifact_path=str(primary_path),
            model_repo_id="primary",
            model_pool=LlamaCppModelPoolConfig(
                max_loaded_models=1,
                models={
                    "primary": LlamaCppModelRegistration(
                        model_id="primary",
                        artifact_path=str(primary_path),
                        base_url="http://127.0.0.1:32210",
                        pinned=False,
                    ),
                    "secondary": LlamaCppModelRegistration(
                        model_id="secondary",
                        artifact_path=str(secondary_path),
                        base_url="http://127.0.0.1:32211",
                        pinned=False,
                    ),
                },
            ),
        ),
        models={
            "primary": ModelProfile(model_id="primary", is_default=True),
            "secondary": ModelProfile(model_id="secondary"),
        },
    )
    pool = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)

    _install_pool_adapter_fakes(pool, "primary", _fake_pool_request("primary", ["turn1"]))
    _install_pool_adapter_fakes(pool, "secondary", _fake_pool_request("secondary", ["turn1"]))

    pool.load_model("primary", reason="manual_load")
    pool._models["primary"].last_used_at = 1.0
    pool.load_model("secondary", reason="manual_load")

    diagnostics = pool.model_pool_diagnostics()
    primary_state = next(
        model for model in diagnostics["models"] if model["model_id"] == "primary"
    )
    secondary_state = next(
        model for model in diagnostics["models"] if model["model_id"] == "secondary"
    )

    assert primary_state["loaded"] is False
    assert primary_state["last_eviction_reason"] == "lru_eviction"
    assert secondary_state["loaded"] is True


def test_llama_cpp_model_pool_applies_ttl_and_idle_unload(tmp_path: Path):
    primary_path = tmp_path / "Qwen3.5-4B-Q4_K_M.gguf"
    secondary_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    primary_path.write_bytes(b"GGUF")
    secondary_path.write_bytes(b"GGUF")

    runtime_config = DGXRuntimeConfig(
        backend=BackendConfig(
            kind="llama_cpp",
            base_url="http://127.0.0.1:32220",
            quant_mode="gguf_experimental",
            model_source="gguf",
            artifact_path=str(primary_path),
            model_repo_id="primary",
            model_pool=LlamaCppModelPoolConfig(
                max_loaded_models=2,
                models={
                    "primary": LlamaCppModelRegistration(
                        model_id="primary",
                        artifact_path=str(primary_path),
                        base_url="http://127.0.0.1:32220",
                        pinned=False,
                        idle_unload_seconds=1,
                    ),
                    "secondary": LlamaCppModelRegistration(
                        model_id="secondary",
                        artifact_path=str(secondary_path),
                        base_url="http://127.0.0.1:32221",
                        pinned=False,
                        ttl_seconds=1,
                    ),
                },
            ),
        ),
        models={
            "primary": ModelProfile(model_id="primary", is_default=True),
            "secondary": ModelProfile(model_id="secondary"),
        },
    )
    pool = LlamaCppModelPoolAdapter.from_runtime_config(runtime_config, tmp_path)

    _install_pool_adapter_fakes(pool, "primary", _fake_pool_request("primary", ["turn1"]))
    _install_pool_adapter_fakes(pool, "secondary", _fake_pool_request("secondary", ["turn1"]))

    pool.load_model("secondary", reason="manual_load")
    pool._models["secondary"].loaded_at = 0.0
    pool._models["secondary"].last_used_at = 0.0
    diagnostics = pool.model_pool_diagnostics()
    secondary_state = next(
        model for model in diagnostics["models"] if model["model_id"] == "secondary"
    )

    pool.load_model("primary", reason="manual_load")
    pool._models["primary"].last_used_at = 0.0
    diagnostics = pool.model_pool_diagnostics()
    primary_state = next(
        model for model in diagnostics["models"] if model["model_id"] == "primary"
    )

    assert secondary_state["loaded"] is False
    assert secondary_state["last_unload_reason"] == "ttl_expired"
    assert primary_state["loaded"] is False
    assert primary_state["last_unload_reason"] == "idle_timeout"
