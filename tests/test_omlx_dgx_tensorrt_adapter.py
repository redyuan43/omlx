# SPDX-License-Identifier: Apache-2.0

import importlib
import importlib.util
import sys
import time
import types
from importlib.machinery import ModuleSpec
from pathlib import Path

import pytest

from omlx_dgx.config import BackendConfig
from omlx_dgx.runtime.backend import BackendError
from omlx_dgx.runtime.tensorrt_llm import TensorRTLLMBackendAdapter


def test_tensorrt_adapter_reports_missing_runtime(tmp_path: Path):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name in {"tensorrt_llm", "tensorrt_llm._tensorrt_engine"}:
            return None
        return original_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = fake_find_spec
    try:
        engine_dir = tmp_path / "engine"
        engine_dir.mkdir()
        (engine_dir / "config.json").write_text("{}", encoding="utf-8")
        config = BackendConfig(
            kind="tensorrt_llm",
            base_url="http://127.0.0.1:65530",
            engine_dir=str(engine_dir),
            launcher_cmd="",
        )

        adapter = TensorRTLLMBackendAdapter.from_backend_config(config, tmp_path)
        diagnostics = adapter.diagnostics().to_dict()

        assert diagnostics["adapter"] == "tensorrt_llm"
        assert diagnostics["engine_dir_exists"] is True
        assert "config.json" in diagnostics["engine_artifacts"]
        assert diagnostics["python_package_available"] is False
        assert diagnostics["engine_module_available"] is False
    finally:
        importlib.util.find_spec = original_find_spec


def test_tensorrt_adapter_logs_when_not_started(tmp_path: Path):
    config = BackendConfig(kind="tensorrt_llm", base_url="http://127.0.0.1:65530")
    adapter = TensorRTLLMBackendAdapter.from_backend_config(config, tmp_path)
    logs = adapter.runtime_logs()

    assert logs["lines"] == []


def test_tensorrt_adapter_can_manage_process_lifecycle(tmp_path: Path):
    config = BackendConfig(
        kind="tensorrt_llm",
        base_url="http://127.0.0.1:65530",
        launcher_cmd="python3 -c \"import time; print('runtime-started', flush=True); time.sleep(5)\"",
        startup_timeout_seconds=1,
        direct_api_enabled=False,
    )
    adapter = TensorRTLLMBackendAdapter.from_backend_config(config, tmp_path)

    started = adapter.start_runtime()
    assert "pid" in started
    time.sleep(0.2)

    logs = adapter.runtime_logs()
    assert any("runtime-started" in line for line in logs["lines"])

    stopped = adapter.stop_runtime()
    assert stopped["stopped"] is True


def test_tensorrt_adapter_can_use_direct_python_api(tmp_path: Path, monkeypatch):
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            assert tokenize is False
            assert add_generation_prompt is True
            return "\n".join(f"{msg['role']}: {msg['content']}" for msg in messages) + "\nassistant:"

    class FakeGenerated:
        text = "pong"
        finish_reason = "stop"
        token_ids = [1, 2]

    class FakeRequestOutput:
        outputs = [FakeGenerated()]
        prompt_token_ids = [10, 11, 12]

    class FakeLLM:
        def __init__(self, model):
            self.model = model
            self.tokenizer = FakeTokenizer()
            self.shutdown_called = False

        def generate(self, prompts, sampling_params=None):
            assert prompts == ["user: Reply with exactly: pong\nassistant:"]
            assert sampling_params.kwargs["max_tokens"] == 8
            return [FakeRequestOutput()]

        def shutdown(self):
            self.shutdown_called = True

    fake_module = types.ModuleType("tensorrt_llm")
    fake_module.__spec__ = ModuleSpec("tensorrt_llm", loader=None)
    fake_module.SamplingParams = FakeSamplingParams
    fake_engine_module = types.ModuleType("tensorrt_llm._tensorrt_engine")
    fake_engine_module.__spec__ = ModuleSpec("tensorrt_llm._tensorrt_engine", loader=None)
    fake_engine_module.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "tensorrt_llm", fake_module)
    monkeypatch.setitem(sys.modules, "tensorrt_llm._tensorrt_engine", fake_engine_module)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name, *args, **kwargs):
        if name == "tensorrt_llm":
            return fake_module.__spec__
        if name == "tensorrt_llm._tensorrt_engine":
            return fake_engine_module.__spec__
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    config = BackendConfig(
        kind="tensorrt_llm",
        base_url="http://127.0.0.1:65530",
        model_repo_id="qwen/qwen3.5-35b-a3b",
        direct_api_enabled=True,
    )
    adapter = TensorRTLLMBackendAdapter.from_backend_config(config, tmp_path)

    started = adapter.start_runtime()
    assert started["mode"] == "direct_api"
    assert adapter.health() is True

    response = adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35",
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            "max_tokens": 8,
            "stream": False,
        },
    )
    payload = response.json()
    assert payload["choices"][0]["message"]["content"] == "pong"
    assert payload["usage"]["prompt_tokens"] == 3

    stream_response = adapter.proxy(
        "POST",
        "v1/chat/completions",
        json={
            "model": "qwen35",
            "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
            "max_tokens": 8,
            "stream": True,
        },
    )
    lines = [chunk.decode("utf-8").strip() for chunk in stream_response.iter_content()]
    assert lines[-1] == "data: [DONE]"

    logs = adapter.runtime_logs()
    assert logs["mode"] == "direct_api"

    stopped = adapter.stop_runtime()
    assert stopped["mode"] == "direct_api"


def test_tensorrt_adapter_surfaces_import_errors(tmp_path: Path, monkeypatch):
    fake_spec = ModuleSpec("tensorrt_llm", loader=None)
    fake_engine_spec = ModuleSpec("tensorrt_llm._tensorrt_engine", loader=None)
    original_find_spec = importlib.util.find_spec
    original_import_module = importlib.import_module

    def fake_find_spec(name, *args, **kwargs):
        if name == "tensorrt_llm":
            return fake_spec
        if name == "tensorrt_llm._tensorrt_engine":
            return fake_engine_spec
        return original_find_spec(name, *args, **kwargs)

    def fake_import_module(name, package=None):
        if name == "tensorrt_llm":
            raise ImportError("libc10_cuda.so: cannot open shared object file")
        return original_import_module(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    config = BackendConfig(
        kind="tensorrt_llm",
        base_url="http://127.0.0.1:65530",
        model_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
        direct_api_enabled=True,
    )
    adapter = TensorRTLLMBackendAdapter.from_backend_config(config, tmp_path)

    diagnostics = adapter.diagnostics().to_dict()
    assert diagnostics["python_package_available"] is True
    assert diagnostics["python_package_importable"] is False
    assert diagnostics["engine_module_available"] is True
    assert "libc10_cuda.so" in diagnostics["direct_api_import_error"]

    with pytest.raises(BackendError, match="failed to import tensorrt_llm direct API"):
        adapter.start_runtime()


def test_tensorrt_adapter_rejects_qwen35_preflight(tmp_path: Path):
    model_dir = tmp_path / "qwen35"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        """
        {
          "architectures": ["Qwen3_5ForConditionalGeneration"],
          "model_type": "qwen3_5",
          "text_config": {"model_type": "qwen3_5_text"}
        }
        """.strip(),
        encoding="utf-8",
    )

    config = BackendConfig(
        kind="tensorrt_llm",
        base_url="http://127.0.0.1:65530",
        model_repo_id=str(model_dir),
        direct_api_enabled=True,
    )
    adapter = TensorRTLLMBackendAdapter.from_backend_config(config, tmp_path)

    diagnostics = adapter.diagnostics().to_dict()
    assert diagnostics["model_architecture"] == "Qwen3_5ForConditionalGeneration"
    assert diagnostics["model_type"] == "qwen3_5"
    assert diagnostics["text_model_type"] == "qwen3_5_text"
    assert diagnostics["model_supported"] is False
    assert "Qwen3.5 is not supported" in diagnostics["unsupported_model_reason"]

    with pytest.raises(BackendError, match="Qwen3.5 is not supported"):
        adapter.start_runtime()
