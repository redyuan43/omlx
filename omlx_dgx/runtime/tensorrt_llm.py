# SPDX-License-Identifier: Apache-2.0
"""First-pass TensorRT-LLM runtime adapter with process management and diagnostics."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import json
import os
import shlex
import shutil
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from huggingface_hub import hf_hub_download

from omlx_dgx.config import BackendConfig

from .backend import BackendError, HttpOpenAIBackendAdapter, RuntimeMetrics


@dataclass
class TensorRTLLMDiagnostics:
    adapter: str
    base_url: str
    direct_api_enabled: bool
    direct_api_active: bool
    runtime_python: str
    runtime_python_exists: bool
    python_package_available: bool
    python_package_importable: bool
    engine_module_available: bool
    tensorrt_package_available: bool
    torch_package_version: str
    tensorrt_package_version: str
    model_architecture: str
    model_type: str
    text_model_type: str
    model_supported: bool
    unsupported_model_reason: str
    launcher_cmd: str
    engine_dir: str
    engine_dir_exists: bool
    engine_artifacts: List[str]
    managed_pid: Optional[int]
    managed_process_running: bool
    startup_timeout_seconds: int
    direct_api_import_error: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TensorRTLLMProcessManager:
    """Manages a configurable TensorRT-LLM launcher command."""

    def __init__(self, state_dir: Path, launcher_cmd: str) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.launcher_cmd = launcher_cmd.strip()
        self.pid_path = self.state_dir / "tensorrt_llm.pid"
        self.log_path = self.state_dir / "tensorrt_llm.log"

    def _read_pid(self) -> Optional[int]:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def is_running(self) -> bool:
        pid = self._read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def pid(self) -> Optional[int]:
        if self.is_running():
            return self._read_pid()
        return None

    def start(self) -> Dict[str, Any]:
        if not self.launcher_cmd:
            raise BackendError("launcher_cmd is not configured")
        if self.is_running():
            return {"started": False, "pid": self.pid(), "reason": "already_running"}

        args = shlex.split(self.launcher_cmd)
        with self.log_path.open("ab") as log_file:
            process = subprocess.Popen(  # noqa: S603
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.pid_path.write_text(str(process.pid), encoding="utf-8")
        return {"started": True, "pid": process.pid}

    def stop(self) -> Dict[str, Any]:
        pid = self.pid()
        if pid is None:
            return {"stopped": False, "reason": "not_running"}
        os.killpg(pid, signal.SIGTERM)
        if self.pid_path.exists():
            self.pid_path.unlink()
        return {"stopped": True, "pid": pid}

    def logs(self, lines: int = 40) -> Dict[str, Any]:
        if not self.log_path.exists():
            return {"lines": [], "path": str(self.log_path)}
        content = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return {"lines": content[-lines:], "path": str(self.log_path)}


@dataclass
class _LocalGeneration:
    text: str
    finish_reason: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class _LocalJSONResponse:
    """Lightweight response object compatible with the current control-plane."""

    def __init__(
        self,
        payload: Dict[str, Any],
        *,
        status_code: int = 200,
        stream_lines: Optional[List[str]] = None,
    ) -> None:
        self._payload = payload
        self.status_code = status_code
        self._stream_lines = stream_lines or []
        self.headers = {
            "content-type": (
                "text/event-stream"
                if self._stream_lines
                else "application/json"
            )
        }

    def json(self) -> Dict[str, Any]:
        return self._payload

    def iter_content(self, chunk_size: int = 8192):
        for line in self._stream_lines:
            yield f"{line}\n\n".encode("utf-8")


class TensorRTLLMDirectRunner:
    """Direct Python LLM API binding for TensorRT-LLM."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self._llm = None
        self._sampling_params_cls = None
        self._tokenizer = None
        self._model_name = config.model_repo_id or config.engine_dir or "tensorrt-llm"

    @property
    def is_started(self) -> bool:
        return self._llm is not None

    def _load_module(self, module_name: str):
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            raise BackendError(
                f"failed to import {module_name} direct API: "
                f"{exc.__class__.__name__}: {exc}"
            ) from exc

    def _load_llm_symbols(self):
        module = self._load_module("tensorrt_llm")
        engine_module = self._load_module("tensorrt_llm._tensorrt_engine")
        llm_cls = getattr(engine_module, "LLM", None)
        sampling_cls = getattr(module, "SamplingParams", None)
        if llm_cls is None or sampling_cls is None:
            raise BackendError(
                "TensorRT-LLM direct API symbols are unavailable: "
                "expected tensorrt_llm._tensorrt_engine.LLM and tensorrt_llm.SamplingParams"
            )
        return llm_cls, sampling_cls

    def _load_model_config(self) -> Optional[Dict[str, Any]]:
        model_ref = self.config.model_repo_id or self.config.engine_dir
        if not model_ref:
            return None

        config_path = Path(model_ref).expanduser() / "config.json"
        if config_path.exists():
            return json.loads(config_path.read_text(encoding="utf-8"))

        try:
            config_file = hf_hub_download(
                repo_id=model_ref,
                filename="config.json",
            )
        except Exception:
            return None
        return json.loads(Path(config_file).read_text(encoding="utf-8"))

    def model_preflight(self) -> Dict[str, Any]:
        config = self._load_model_config()
        if not config:
            return {
                "architecture": "",
                "model_type": "",
                "text_model_type": "",
                "supported": True,
                "reason": "",
            }

        architectures = config.get("architectures") or []
        architecture = architectures[0] if architectures else ""
        model_type = str(config.get("model_type", ""))
        text_model_type = str((config.get("text_config") or {}).get("model_type", ""))

        if (
            architecture == "Qwen3_5ForConditionalGeneration"
            or model_type == "qwen3_5"
            or text_model_type == "qwen3_5_text"
        ):
            return {
                "architecture": architecture,
                "model_type": model_type,
                "text_model_type": text_model_type,
                "supported": False,
                "reason": (
                    "Qwen3.5 is not supported by the installed TensorRT-LLM 1.1.0 runtime. "
                    "The checkpoint uses the qwen3_5 architecture with text/vision structure "
                    "that is not present in the current TensorRT-LLM model map."
                ),
            }

        return {
            "architecture": architecture,
            "model_type": model_type,
            "text_model_type": text_model_type,
            "supported": True,
            "reason": "",
        }

    def start(self) -> Dict[str, Any]:
        if self._llm is not None:
            return {"started": False, "mode": "direct_api", "reason": "already_running"}

        os.environ.setdefault("TLLM_USE_TRT_ENGINE", "1")

        llm_cls, sampling_cls = self._load_llm_symbols()

        model_ref = self.config.model_repo_id or self.config.engine_dir
        if not model_ref:
            raise BackendError("model_repo_id or engine_dir must be configured for direct API")

        preflight = self.model_preflight()
        if not preflight["supported"]:
            raise BackendError(preflight["reason"])

        self._llm = llm_cls(model=model_ref)
        self._sampling_params_cls = sampling_cls
        self._tokenizer = getattr(self._llm, "tokenizer", None)
        return {
            "started": True,
            "mode": "direct_api",
            "model": self._model_name,
        }

    def stop(self) -> Dict[str, Any]:
        if self._llm is None:
            return {"stopped": False, "mode": "direct_api", "reason": "not_running"}

        shutdown = getattr(self._llm, "shutdown", None)
        if callable(shutdown):
            shutdown()
        self._llm = None
        self._sampling_params_cls = None
        self._tokenizer = None
        return {"stopped": True, "mode": "direct_api"}

    def _ensure_started(self) -> None:
        if self._llm is None:
            self.start()

    def _chat_prompt(self, messages: List[Dict[str, Any]]) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass

        lines: List[str] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "")
                    if isinstance(item, dict)
                    else str(item)
                    for item in content
                )
            lines.append(f"{role}: {content}")
        lines.append("assistant:")
        return "\n".join(lines)

    def _sampling_params(self, payload: Dict[str, Any]):
        self._ensure_started()
        assert self._sampling_params_cls is not None
        kwargs = {
            "max_tokens": payload.get("max_tokens"),
            "temperature": payload.get("temperature"),
            "top_p": payload.get("top_p"),
            "top_k": payload.get("top_k"),
            "stop": payload.get("stop"),
        }
        filtered = {key: value for key, value in kwargs.items() if value is not None}
        return self._sampling_params_cls(**filtered)

    def _first_generation(self, prompts: List[str], payload: Dict[str, Any]) -> _LocalGeneration:
        self._ensure_started()
        assert self._llm is not None

        sampling_params = self._sampling_params(payload)
        outputs = self._llm.generate(prompts, sampling_params=sampling_params)
        first_output = outputs[0]
        generated = first_output.outputs[0]
        completion_tokens = None
        prompt_tokens = None
        token_ids = getattr(generated, "token_ids", None)
        if token_ids is not None:
            completion_tokens = len(token_ids)
        prompt_token_ids = getattr(first_output, "prompt_token_ids", None)
        if prompt_token_ids is not None:
            prompt_tokens = len(prompt_token_ids)
        return _LocalGeneration(
            text=getattr(generated, "text", ""),
            finish_reason=getattr(generated, "finish_reason", "stop"),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def chat_response(self, payload: Dict[str, Any]) -> _LocalJSONResponse:
        prompt = self._chat_prompt(payload.get("messages", []))
        generation = self._first_generation([prompt], payload)
        created = int(time.time())
        response_payload = {
            "id": f"chatcmpl-direct-{created}",
            "object": "chat.completion",
            "created": created,
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generation.text,
                        "tool_calls": [],
                    },
                    "logprobs": None,
                    "finish_reason": generation.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": (
                    (generation.prompt_tokens or 0) + (generation.completion_tokens or 0)
                ),
            },
            "stats": {"backend": "tensorrt_llm_direct"},
        }
        if payload.get("stream"):
            stream_lines = [
                "data: "
                + json.dumps(
                    {
                        "id": response_payload["id"],
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self._model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "content": generation.text,
                                },
                                "logprobs": None,
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
                "data: "
                + json.dumps(
                    {
                        "id": response_payload["id"],
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": self._model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "logprobs": None,
                                "finish_reason": generation.finish_reason,
                            }
                        ],
                    }
                ),
                "data: [DONE]",
            ]
            return _LocalJSONResponse(response_payload, stream_lines=stream_lines)

        return _LocalJSONResponse(response_payload)

    def completion_response(self, payload: Dict[str, Any]) -> _LocalJSONResponse:
        prompt = payload.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "\n".join(str(item) for item in prompt)
        generation = self._first_generation([str(prompt)], payload)
        created = int(time.time())
        response_payload = {
            "id": f"cmpl-direct-{created}",
            "object": "text_completion",
            "created": created,
            "model": self._model_name,
            "choices": [
                {
                    "index": 0,
                    "text": generation.text,
                    "logprobs": None,
                    "finish_reason": generation.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": generation.prompt_tokens,
                "completion_tokens": generation.completion_tokens,
                "total_tokens": (
                    (generation.prompt_tokens or 0) + (generation.completion_tokens or 0)
                ),
            },
            "stats": {"backend": "tensorrt_llm_direct"},
        }
        if payload.get("stream"):
            stream_lines = [
                "data: "
                + json.dumps(
                    {
                        "id": response_payload["id"],
                        "object": "text_completion",
                        "created": created,
                        "model": self._model_name,
                        "choices": [
                            {
                                "index": 0,
                                "text": generation.text,
                                "logprobs": None,
                                "finish_reason": generation.finish_reason,
                            }
                        ],
                    }
                ),
                "data: [DONE]",
            ]
            return _LocalJSONResponse(response_payload, stream_lines=stream_lines)

        return _LocalJSONResponse(response_payload)


class TensorRTLLMBackendAdapter(HttpOpenAIBackendAdapter):
    """Runtime-aware adapter for TensorRT-LLM-style backends."""

    def __init__(self, config: BackendConfig, state_dir: Path) -> None:
        super().__init__(config.base_url)
        self.config = config
        self.state_dir = Path(state_dir)
        self.process_manager = TensorRTLLMProcessManager(
            self.state_dir / "runtime",
            config.launcher_cmd,
        )
        self.direct_runner = TensorRTLLMDirectRunner(config)

    @staticmethod
    def _package_version(package_name: str) -> str:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return ""

    def _direct_api_probe(self) -> Dict[str, Any]:
        available = importlib.util.find_spec("tensorrt_llm") is not None
        engine_available = importlib.util.find_spec("tensorrt_llm._tensorrt_engine") is not None
        if not available:
            return {
                "available": False,
                "importable": False,
                "has_llm": False,
                "has_sampling_params": False,
                "error": "",
                "engine_available": engine_available,
            }

        try:
            module = importlib.import_module("tensorrt_llm")
        except Exception as exc:
            return {
                "available": True,
                "importable": False,
                "has_llm": False,
                "has_sampling_params": False,
                "error": f"{exc.__class__.__name__}: {exc}",
                "engine_available": engine_available,
            }

        try:
            engine_module = importlib.import_module("tensorrt_llm._tensorrt_engine")
        except Exception as exc:
            return {
                "available": True,
                "importable": False,
                "has_llm": False,
                "has_sampling_params": getattr(module, "SamplingParams", None) is not None,
                "error": f"{exc.__class__.__name__}: {exc}",
                "engine_available": engine_available,
            }

        return {
            "available": True,
            "importable": True,
            "has_llm": getattr(engine_module, "LLM", None) is not None,
            "has_sampling_params": getattr(module, "SamplingParams", None) is not None,
            "error": "",
            "engine_available": engine_available,
        }

    def _direct_api_supported(self) -> bool:
        probe = self._direct_api_probe()
        return (
            self.config.direct_api_enabled
            and probe["importable"]
            and probe["has_llm"]
            and probe["has_sampling_params"]
        )

    def _engine_artifacts(self) -> List[str]:
        if not self.config.engine_dir:
            return []
        engine_dir = Path(self.config.engine_dir).expanduser().resolve()
        if not engine_dir.exists():
            return []
        patterns = [
            "*.engine",
            "*.plan",
            "config.json",
            "engine_config.json",
            "model.cache",
        ]
        found: List[str] = []
        for pattern in patterns:
            for path in sorted(engine_dir.glob(pattern)):
                found.append(path.name)
        return found

    def diagnostics(self) -> TensorRTLLMDiagnostics:
        runtime_python = shutil.which(self.config.runtime_python) or ""
        engine_dir = Path(self.config.engine_dir).expanduser().resolve() if self.config.engine_dir else None
        direct_probe = self._direct_api_probe()
        model_probe = self.direct_runner.model_preflight()
        return TensorRTLLMDiagnostics(
            adapter="tensorrt_llm",
            base_url=self.base_url,
            direct_api_enabled=self.config.direct_api_enabled,
            direct_api_active=self.direct_runner.is_started,
            runtime_python=self.config.runtime_python,
            runtime_python_exists=bool(runtime_python),
            python_package_available=direct_probe["available"],
            python_package_importable=direct_probe["importable"],
            engine_module_available=direct_probe["engine_available"],
            tensorrt_package_available=importlib.util.find_spec("tensorrt") is not None,
            torch_package_version=self._package_version("torch"),
            tensorrt_package_version=self._package_version("tensorrt"),
            model_architecture=model_probe["architecture"],
            model_type=model_probe["model_type"],
            text_model_type=model_probe["text_model_type"],
            model_supported=model_probe["supported"],
            unsupported_model_reason=model_probe["reason"],
            launcher_cmd=self.config.launcher_cmd,
            engine_dir=str(engine_dir) if engine_dir is not None else "",
            engine_dir_exists=bool(engine_dir and engine_dir.exists()),
            engine_artifacts=self._engine_artifacts(),
            managed_pid=self.process_manager.pid(),
            managed_process_running=self.process_manager.is_running(),
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            direct_api_import_error=direct_probe["error"],
        )

    def health(self) -> bool:
        if self.direct_runner.is_started:
            return True
        return super().health()

    def collect_metrics(self) -> RuntimeMetrics:
        metrics = super().collect_metrics()
        metrics.details = self.diagnostics().to_dict()
        return metrics

    def proxy(self, method: str, path: str, **kwargs: Any):
        if self._direct_api_supported() and path in {"v1/chat/completions", "v1/completions"}:
            payload = kwargs.get("json")
            if not isinstance(payload, dict):
                raise BackendError("direct TensorRT-LLM API expects JSON payload")
            if path == "v1/chat/completions":
                return self.direct_runner.chat_response(payload)
            return self.direct_runner.completion_response(payload)
        return super().proxy(method, path, **kwargs)

    def start_runtime(self) -> Dict[str, Any]:
        if self.config.direct_api_enabled:
            probe = self._direct_api_probe()
            if probe["available"] or probe["engine_available"]:
                return self.direct_runner.start()
        result = self.process_manager.start()
        if not result.get("started"):
            return result

        deadline = time.time() + self.config.startup_timeout_seconds
        while time.time() < deadline:
            if self.health():
                result["healthy"] = True
                return result
            time.sleep(1)

        result["healthy"] = False
        result["reason"] = "healthcheck_timeout"
        return result

    def stop_runtime(self) -> Dict[str, Any]:
        if self.direct_runner.is_started:
            return self.direct_runner.stop()
        return self.process_manager.stop()

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        if self.direct_runner.is_started:
            return {
                "lines": [],
                "mode": "direct_api",
                "path": None,
                "diagnostics": self.diagnostics().to_dict(),
            }
        payload = self.process_manager.logs(lines=lines)
        payload["diagnostics"] = self.diagnostics().to_dict()
        return payload

    @classmethod
    def from_backend_config(cls, config: BackendConfig, base_path: Path) -> "TensorRTLLMBackendAdapter":
        return cls(config=config, state_dir=base_path)
