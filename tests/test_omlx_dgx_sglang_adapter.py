# SPDX-License-Identifier: Apache-2.0

import json
import time
from pathlib import Path

import pytest

from omlx_dgx.config import BackendConfig
from omlx_dgx.runtime import sglang as sglang_module
from omlx_dgx.runtime.backend import BackendError
from omlx_dgx.runtime.sglang import SGLangBackendAdapter


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


def test_sglang_adapter_builds_launch_command_and_env(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: True)
    monkeypatch.setattr(
        sglang_module,
        "_detect_runtime_library_path",
        lambda _: "/opt/runtime/lib:/opt/runtime/cuda",
    )
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31000",
        runtime_python="/opt/venvs/sglang/bin/python",
        model_repo_id="Qwen/Qwen3.5-35B-A3B",
        admin_api_key="secret-admin",
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    env = adapter._build_env()

    assert command[:3] == ["/opt/venvs/sglang/bin/python", "-m", "sglang.launch_server"]
    assert "--model-path" in command
    assert "--tp-size" in command
    assert "--context-length" in command
    assert "--chunked-prefill-size" in command
    assert command[command.index("--chunked-prefill-size") + 1] == "8192"
    assert "--dtype" in command
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--chat-template" not in command
    assert "--attention-backend" in command
    assert command[command.index("--attention-backend") + 1] == "triton"
    assert "--reasoning-parser" in command
    assert "--enable-hierarchical-cache" in command
    assert "--hicache-storage-backend" in command
    assert "--admin-api-key" in command
    assert json.loads(
        command[command.index("--hicache-storage-backend-extra-config") + 1]
    ) == {
        "prefetch_threshold": 256,
        "prefetch_timeout_base": 0.5,
        "prefetch_timeout_per_ki_token": 0.25,
        "hicache_storage_pass_prefix_keys": True,
    }
    assert env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] == str(
        Path(config.hicache_storage_root).expanduser().resolve()
    )
    assert env["LD_LIBRARY_PATH"].startswith("/opt/runtime/lib:/opt/runtime/cuda")
    assert env["TRITON_PTXAS_PATH"]

    diagnostics = adapter.diagnostics().to_dict()
    assert diagnostics["adapter"] == "sglang"
    assert diagnostics["backend_format"] == "sglang_bf16"
    assert diagnostics["quant_mode"] == "bf16"
    assert diagnostics["model_source"] == "hf"
    assert diagnostics["effective_model_path"] == "Qwen/Qwen3.5-35B-A3B"
    assert diagnostics["attention_backend"] == "triton"
    assert diagnostics["mamba_ssm_dtype"] == ""
    assert diagnostics["disable_cuda_graph"] is False
    assert diagnostics["chunked_prefill_size"] == 8192
    assert diagnostics["hicache_storage_backend"] == "file"
    assert diagnostics["admin_api_key_configured"] is True
    assert diagnostics["artifact_summary"]["artifact_kind"] == "remote_ref"


def test_sglang_adapter_includes_chat_template_when_configured(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: False)
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31001",
        runtime_python="/opt/venvs/sglang/bin/python",
        model_repo_id="Qwen/Qwen3.5-4B",
        chat_template="custom-template",
        attention_backend="trtllm_mha",
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()

    assert "--chat-template" in command
    assert command[command.index("--chat-template") + 1] == "custom-template"
    assert command[command.index("--attention-backend") + 1] == "trtllm_mha"


def test_sglang_adapter_builds_awq_launch_command(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: False)
    awq_dir = tmp_path / "Qwen3.5-4B-AWQ"
    awq_dir.mkdir()
    (awq_dir / "config.json").write_text(
        json.dumps(
            {
                "torch_dtype": "float16",
                "quantization_config": {"quant_method": "awq", "bits": 4},
            }
        ),
        encoding="utf-8",
    )
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31002",
        model_repo_id="Qwen/Qwen3.5-4B",
        quant_mode="awq_int4",
        artifact_path=str(awq_dir),
        mamba_ssm_dtype="float16",
        disable_cuda_graph=True,
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert command[command.index("--model-path") + 1] == str(awq_dir)
    assert command[command.index("--quantization") + 1] == "awq"
    assert command[command.index("--dtype") + 1] == "half"
    assert command[command.index("--mamba-ssm-dtype") + 1] == "float16"
    assert "--disable-cuda-graph" in command
    assert diagnostics["backend_format"] == "sglang_awq_int4"
    assert diagnostics["mamba_ssm_dtype"] == "float16"
    assert diagnostics["disable_cuda_graph"] is True
    assert diagnostics["artifact_summary"]["artifact_kind"] == "local_dir"
    assert diagnostics["artifact_summary"]["quantization_hint"] == "awq_4bit"


def test_sglang_adapter_builds_awq_marlin_launch_command(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: False)
    awq_dir = tmp_path / "Qwen3.5-4B-AWQ"
    awq_dir.mkdir()
    (awq_dir / "config.json").write_text(
        json.dumps(
            {
                "torch_dtype": "float16",
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 4,
                    "group_size": 128,
                    "zero_point": True,
                },
            }
        ),
        encoding="utf-8",
    )
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31012",
        model_repo_id="Qwen/Qwen3.5-4B",
        quant_mode="awq_marlin_int4",
        artifact_path=str(awq_dir),
        mamba_ssm_dtype="float16",
        disable_cuda_graph=True,
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert command[command.index("--model-path") + 1] == str(awq_dir)
    assert command[command.index("--quantization") + 1] == "awq_marlin"
    assert command[command.index("--dtype") + 1] == "half"
    assert command[command.index("--mamba-ssm-dtype") + 1] == "float16"
    assert "--disable-cuda-graph" in command
    assert diagnostics["backend_format"] == "sglang_awq_marlin_int4"
    assert diagnostics["quantization"] == "awq_marlin"
    assert diagnostics["disable_cuda_graph"] is True
    assert diagnostics["artifact_summary"]["quantization_hint"] == "awq_4bit"


def test_sglang_adapter_builds_gguf_launch_command(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: False)
    gguf_path = tmp_path / "Qwen3.5-4B-Q4_K_S.gguf"
    gguf_path.write_bytes(b"GGUF")
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31003",
        model_repo_id="Qwen/Qwen3.5-4B",
        quant_mode="gguf_experimental",
        artifact_path=str(gguf_path),
        enable_hierarchical_cache=False,
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    command = adapter._build_launch_command()
    diagnostics = adapter.diagnostics().to_dict()

    assert command[command.index("--model-path") + 1] == str(gguf_path)
    assert command[command.index("--load-format") + 1] == "gguf"
    assert command[command.index("--quantization") + 1] == "gguf"
    assert "--dtype" not in command
    assert diagnostics["backend_format"] == "sglang_gguf_experimental"
    assert diagnostics["artifact_summary"]["artifact_kind"] == "local_file"
    assert diagnostics["artifact_summary"]["quantization_hint"] == "gguf"


def test_sglang_adapter_hicache_and_metrics_calls_use_admin_auth(
    tmp_path: Path, monkeypatch
):
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:32000",
        admin_api_key="secret-admin",
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)
    calls = []

    def fake_request(method, url, timeout=0, headers=None, json=None, **kwargs):
        path = url.removeprefix("http://127.0.0.1:32000/")
        calls.append((method, path, headers or {}, json))
        if path in {"health", "v1/models"}:
            return FakeResponse(
                json_data={"object": "list", "data": [{"id": "Qwen/Qwen3.5-35B-A3B"}]},
                headers={"content-type": "application/json"},
            )
        if path in {"server_info", "get_server_info"}:
            return FakeResponse(
                json_data={
                    "enable_cache_report": True,
                    "enable_hierarchical_cache": True,
                    "hicache_storage_backend": "file",
                    "internal_states": [{"cached_tokens": 64}],
                    "page_size": 64,
                },
                headers={"content-type": "application/json"},
            )
        if path == "metrics":
            return FakeResponse(
                text="sglang_requests_total 12\nsglang_cache_hits_total 7\n",
                headers={"content-type": "text/plain"},
            )
        if path == "hicache/storage-backend" and method == "GET":
            assert headers == {"Authorization": "Bearer secret-admin"}
            return FakeResponse(
                json_data={"hicache_storage_backend": "file"},
                headers={"content-type": "application/json"},
            )
        if path == "hicache/storage-backend" and method == "PUT":
            assert headers == {"Authorization": "Bearer secret-admin"}
            return FakeResponse(
                json_data={"attached": True, "payload": json},
                headers={"content-type": "application/json"},
            )
        if path == "hicache/storage-backend" and method == "DELETE":
            assert headers == {"Authorization": "Bearer secret-admin"}
            return FakeResponse(
                json_data={"detached": True},
                headers={"content-type": "application/json"},
            )
        raise AssertionError(f"unexpected request: {method} {path}")

    monkeypatch.setattr(adapter.session, "request", fake_request)

    status = adapter.hicache_storage_status()
    assert status["hicache_storage_backend"] == "file"

    attached = adapter.attach_hicache_storage_backend(
        overrides={"hicache_storage_backend": "dynamic"}
    )
    assert attached["attached"] is True
    assert attached["payload"]["hicache_storage_backend"] == "dynamic"
    assert "hicache_storage_backend_extra_config_json" in attached["payload"]

    detached = adapter.detach_hicache_storage_backend()
    assert detached["detached"] is True

    cache_report = adapter.cache_report()
    assert cache_report["internal_states"][0]["cached_tokens"] == 64

    metrics = adapter.collect_metrics().to_dict()
    assert metrics["details"]["hicache_storage"]["hicache_storage_backend"] == "file"
    assert metrics["details"]["cache_report"]["enable_cache_report"] is True
    assert "sglang_requests_total 12" in metrics["details"]["metrics_excerpt"][0]

    assert any(path == "hicache/storage-backend" and method == "PUT" for method, path, _, _ in calls)


def test_sglang_adapter_surfaces_hicache_incompatibility_for_qwen35(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: False)
    model_dir = tmp_path / "Qwen3.5-4B"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31000",
        runtime_python="/opt/venvs/sglang/bin/python",
        model_repo_id=str(model_dir),
        enable_hierarchical_cache=True,
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    diagnostics = adapter.diagnostics().to_dict()

    assert diagnostics["model_type"] == "qwen3_5"
    assert diagnostics["model_architectures"] == ["Qwen3_5ForConditionalGeneration"]
    assert diagnostics["hicache_supported"] is False
    assert "hybrid GDN/Mamba" in diagnostics["hicache_blocker"]

    with pytest.raises(BackendError, match="HiCache is not supported"):
        adapter.start_runtime()


def test_sglang_adapter_reports_lmstudio_baseline_as_non_runnable(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setattr(sglang_module, "_runtime_python_exists", lambda _: False)
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:31004",
        model_repo_id="Qwen/Qwen3.5-4B",
        quant_mode="lmstudio_baseline",
        model_source="lmstudio_api",
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)

    diagnostics = adapter.diagnostics().to_dict()

    assert diagnostics["backend_format"] == "lmstudio_baseline"
    assert "reporting-only quant_mode" in diagnostics["launcher_cmd_error"]

    with pytest.raises(BackendError, match="reporting-only quant_mode"):
        adapter.start_runtime()


def test_sglang_adapter_can_manage_process_lifecycle(tmp_path: Path, monkeypatch):
    config = BackendConfig(
        kind="sglang",
        base_url="http://127.0.0.1:33000",
        launcher_cmd="python3 -c \"import time; print('sglang-started', flush=True); time.sleep(5)\"",
        startup_timeout_seconds=2,
        enable_hierarchical_cache=False,
    )
    adapter = SGLangBackendAdapter.from_backend_config(config, tmp_path)
    health_checks = {"count": 0}

    def fake_health() -> bool:
        health_checks["count"] += 1
        return health_checks["count"] >= 2

    monkeypatch.setattr(adapter, "health", fake_health)

    started = adapter.start_runtime()
    assert started["mode"] == "sglang"
    time.sleep(0.2)

    logs = adapter.runtime_logs()
    assert any("sglang-started" in line for line in logs["lines"])

    stopped = adapter.stop_runtime()
    assert stopped["stopped"] is True
