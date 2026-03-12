# SPDX-License-Identifier: Apache-2.0
"""SGLang runtime adapter with HiCache-aware process management."""

from __future__ import annotations

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
from urllib.parse import urlparse

from omlx_dgx.config import BackendConfig

from .backend import BackendError, HttpOpenAIBackendAdapter, RuntimeMetrics


def _stringify_command(args: List[str]) -> str:
    return shlex.join(args)


def _runtime_python_exists(runtime_python: str) -> bool:
    if "/" in runtime_python:
        return Path(runtime_python).expanduser().exists()
    return shutil.which(runtime_python) is not None


def _detect_runtime_library_path(runtime_python: str) -> str:
    probe = (
        "from pathlib import Path; "
        "import torch; "
        "root = Path(torch.__file__).resolve().parent; "
        "cands = ["
        "root / 'lib', "
        "root.parent / 'nvidia' / 'cu12' / 'lib', "
        "root.parent / 'nvidia' / 'cu13' / 'lib', "
        "root.parent / 'nvidia' / 'cuda_nvrtc' / 'lib', "
        "root.parent / 'nvidia' / 'cuda_runtime' / 'lib', "
        "root.parent / 'nvidia' / 'cublas' / 'lib', "
        "root.parent / 'nvidia' / 'cudnn' / 'lib', "
        "root.parent / 'nvidia' / 'cusolver' / 'lib', "
        "root.parent / 'nvidia' / 'cusparse' / 'lib', "
        "root.parent / 'nvidia' / 'nccl' / 'lib', "
        "root.parent / 'nvidia' / 'nvjitlink' / 'lib'"
        "]; "
        "print(':'.join(str(p) for p in cands if p.is_dir()))"
    )
    result = subprocess.run(
        [runtime_python, "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _coerce_response_payload(response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "text": response.text.strip(),
    }


def _latest_model_config_path(model_repo_id: str) -> Optional[Path]:
    candidate = Path(model_repo_id).expanduser()
    if candidate.is_dir():
        config_path = candidate / "config.json"
        if config_path.exists():
            return config_path

    cache_root = (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / f"models--{model_repo_id.replace('/', '--')}"
        / "snapshots"
    )
    if not cache_root.exists():
        return None

    configs = sorted(
        cache_root.glob("*/config.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return configs[0] if configs else None


def _inspect_model_traits(model_repo_id: str) -> Dict[str, Any]:
    traits: Dict[str, Any] = {
        "config_path": "",
        "model_type": "",
        "architectures": [],
        "hicache_supported": None,
        "hicache_blocker": "",
    }
    config_path = _latest_model_config_path(model_repo_id)
    if config_path is None:
        return traits

    traits["config_path"] = str(config_path)
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        traits["hicache_blocker"] = f"failed to parse model config: {exc}"
        return traits

    architectures = config.get("architectures") or []
    model_type = config.get("model_type") or ""
    traits["architectures"] = architectures
    traits["model_type"] = model_type

    if model_type.startswith("qwen3_5") or "Qwen3_5ForConditionalGeneration" in architectures:
        traits["hicache_supported"] = False
        traits["hicache_blocker"] = (
            "installed SGLang HiCache only supports MHA/MLA caches, while "
            "Qwen3.5 uses the hybrid GDN/Mamba cache path"
        )

    return traits


@dataclass
class SGLangDiagnostics:
    adapter: str
    base_url: str
    runtime_python: str
    runtime_python_exists: bool
    sglang_package_available: bool
    sgl_kernel_package_available: bool
    sglang_package_version: str
    sgl_kernel_package_version: str
    model_repo_id: str
    model_config_path: str
    model_type: str
    model_architectures: List[str]
    tensor_parallel_size: int
    context_length: int
    chunked_prefill_size: int
    chat_template: str
    attention_backend: str
    reasoning_parser: str
    mem_fraction_static: float
    enable_metrics: bool
    enable_cache_report: bool
    enable_hierarchical_cache: bool
    hicache_storage_backend: str
    hicache_storage_root: str
    hicache_supported: Optional[bool]
    hicache_blocker: str
    launcher_cmd: str
    managed_pid: Optional[int]
    managed_process_running: bool
    startup_timeout_seconds: int
    admin_api_key_configured: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SGLangProcessManager:
    """Manages a configurable SGLang launcher command."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.state_dir / "sglang.pid"
        self.log_path = self.state_dir / "sglang.log"
        self.cmd_path = self.state_dir / "sglang.cmd"

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

    def start(self, args: List[str], *, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if self.is_running():
            return {"started": False, "pid": self.pid(), "reason": "already_running"}

        with self.log_path.open("ab") as log_file:
            process = subprocess.Popen(  # noqa: S603
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.pid_path.write_text(str(process.pid), encoding="utf-8")
        self.cmd_path.write_text(_stringify_command(args), encoding="utf-8")
        return {
            "started": True,
            "pid": process.pid,
            "command": _stringify_command(args),
        }

    def stop(self) -> Dict[str, Any]:
        pid = self.pid()
        if pid is None:
            return {"stopped": False, "reason": "not_running"}
        os.killpg(pid, signal.SIGTERM)
        if self.pid_path.exists():
            self.pid_path.unlink()
        return {"stopped": True, "pid": pid}

    def logs(self, lines: int = 40) -> Dict[str, Any]:
        content: List[str] = []
        if self.log_path.exists():
            content = self.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        return {
            "lines": content[-lines:],
            "path": str(self.log_path),
            "command": self.command(),
        }

    def command(self) -> str:
        if not self.cmd_path.exists():
            return ""
        return self.cmd_path.read_text(encoding="utf-8").strip()


class SGLangBackendAdapter(HttpOpenAIBackendAdapter):
    """Runtime-aware adapter for SGLang OpenAI-compatible backends."""

    def __init__(self, config: BackendConfig, state_dir: Path) -> None:
        super().__init__(config.base_url, timeout=120.0)
        self.config = config
        self.state_dir = Path(state_dir)
        self.process_manager = SGLangProcessManager(self.state_dir / "runtime")
        parsed = urlparse(config.base_url)
        self.launch_host = parsed.hostname or "127.0.0.1"
        self.launch_port = parsed.port or 30000

    @classmethod
    def from_backend_config(
        cls, config: BackendConfig, state_dir: str | Path
    ) -> "SGLangBackendAdapter":
        return cls(config, Path(state_dir))

    @staticmethod
    def _package_version(package_name: str) -> str:
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return ""

    def _build_hicache_extra_config(self) -> Dict[str, Any]:
        return dict(self.config.hicache_storage_backend_extra_config)

    def _build_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        if _runtime_python_exists(self.config.runtime_python):
            try:
                runtime_library_path = _detect_runtime_library_path(
                    self.config.runtime_python
                )
            except Exception:
                runtime_library_path = ""
            if runtime_library_path:
                env["LD_LIBRARY_PATH"] = (
                    runtime_library_path
                    + (
                        f":{env['LD_LIBRARY_PATH']}"
                        if env.get("LD_LIBRARY_PATH")
                        else ""
                    )
                )
        if "TRITON_PTXAS_PATH" not in env:
            ptxas = shutil.which("ptxas")
            if ptxas:
                env["TRITON_PTXAS_PATH"] = ptxas
        if self.config.hicache_storage_backend == "file":
            storage_root = Path(self.config.hicache_storage_root).expanduser().resolve()
            storage_root.mkdir(parents=True, exist_ok=True)
            env["SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"] = str(storage_root)
        return env

    def _build_launch_command(self) -> List[str]:
        if self.config.launcher_cmd:
            return shlex.split(self.config.launcher_cmd)

        if not self.config.model_repo_id:
            raise BackendError("model_repo_id is required for SGLang runtime startup")

        args = [
            self.config.runtime_python,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.config.model_repo_id,
            "--host",
            self.launch_host,
            "--port",
            str(self.launch_port),
            "--tp-size",
            str(self.config.tensor_parallel_size),
            "--context-length",
            str(self.config.context_length),
            "--chunked-prefill-size",
            str(self.config.chunked_prefill_size),
            "--mem-fraction-static",
            str(self.config.mem_fraction_static),
        ]
        if self.config.chat_template:
            args.extend(["--chat-template", self.config.chat_template])
        if self.config.attention_backend:
            args.extend(["--attention-backend", self.config.attention_backend])
        if self.config.reasoning_parser:
            args.extend(["--reasoning-parser", self.config.reasoning_parser])
        if self.config.trust_remote_code:
            args.append("--trust-remote-code")
        if self.config.enable_metrics:
            args.append("--enable-metrics")
        if self.config.enable_cache_report:
            args.append("--enable-cache-report")
        if self.config.enable_hierarchical_cache:
            args.extend(
                [
                    "--enable-hierarchical-cache",
                    "--page-size",
                    str(self.config.page_size),
                    "--hicache-mem-layout",
                    self.config.hicache_mem_layout,
                    "--hicache-io-backend",
                    self.config.hicache_io_backend,
                    "--hicache-write-policy",
                    self.config.hicache_write_policy,
                    "--hicache-storage-backend",
                    self.config.hicache_storage_backend,
                    "--hicache-storage-prefetch-policy",
                    self.config.hicache_storage_prefetch_policy,
                ]
            )
            if self.config.hicache_size > 0:
                args.extend(["--hicache-size", str(self.config.hicache_size)])
            else:
                args.extend(["--hicache-ratio", str(self.config.hicache_ratio)])

            extra_config = self._build_hicache_extra_config()
            if extra_config:
                args.extend(
                    [
                        "--hicache-storage-backend-extra-config",
                        json.dumps(extra_config, sort_keys=True),
                    ]
                )
        if self.config.admin_api_key:
            args.extend(["--admin-api-key", self.config.admin_api_key])
        return args

    def diagnostics(self) -> SGLangDiagnostics:
        command = self._build_launch_command()
        model_traits = _inspect_model_traits(self.config.model_repo_id)
        return SGLangDiagnostics(
            adapter="sglang",
            base_url=self.base_url,
            runtime_python=self.config.runtime_python,
            runtime_python_exists=_runtime_python_exists(self.config.runtime_python),
            sglang_package_available=importlib.util.find_spec("sglang") is not None,
            sgl_kernel_package_available=importlib.util.find_spec("sgl_kernel") is not None,
            sglang_package_version=self._package_version("sglang"),
            sgl_kernel_package_version=self._package_version("sgl-kernel"),
            model_repo_id=self.config.model_repo_id,
            model_config_path=model_traits["config_path"],
            model_type=model_traits["model_type"],
            model_architectures=list(model_traits["architectures"]),
            tensor_parallel_size=self.config.tensor_parallel_size,
            context_length=self.config.context_length,
            chunked_prefill_size=self.config.chunked_prefill_size,
            chat_template=self.config.chat_template or "",
            attention_backend=self.config.attention_backend,
            reasoning_parser=self.config.reasoning_parser,
            mem_fraction_static=self.config.mem_fraction_static,
            enable_metrics=self.config.enable_metrics,
            enable_cache_report=self.config.enable_cache_report,
            enable_hierarchical_cache=self.config.enable_hierarchical_cache,
            hicache_storage_backend=self.config.hicache_storage_backend,
            hicache_storage_root=str(
                Path(self.config.hicache_storage_root).expanduser().resolve()
            ),
            hicache_supported=model_traits["hicache_supported"],
            hicache_blocker=model_traits["hicache_blocker"],
            launcher_cmd=_stringify_command(command),
            managed_pid=self.process_manager.pid(),
            managed_process_running=self.process_manager.is_running(),
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            admin_api_key_configured=bool(self.config.admin_api_key),
        )

    def _admin_headers(self) -> Dict[str, str]:
        if not self.config.admin_api_key:
            return {}
        return {"Authorization": f"Bearer {self.config.admin_api_key}"}

    def _server_info(self) -> Dict[str, Any]:
        for path in ("server_info", "get_server_info"):
            try:
                response = self._request("GET", path, timeout=10)
            except Exception:
                continue
            if response.ok:
                return response.json()
        return {}

    def _metrics_excerpt(self, max_lines: int = 20) -> List[str]:
        try:
            response = self._request("GET", "metrics", timeout=10)
        except Exception:
            return []
        if not response.ok:
            return []
        return response.text.splitlines()[:max_lines]

    def _request_admin(self, method: str, path: str, **kwargs: Any):
        headers = dict(kwargs.pop("headers", {}))
        headers.update(self._admin_headers())
        try:
            return self._request(method, path, headers=headers, **kwargs)
        except Exception as exc:
            raise BackendError(str(exc)) from exc

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.health():
                return
            if not self.process_manager.is_running():
                logs = self.process_manager.logs(lines=40)["lines"]
                log_tail = "\n".join(logs)
                raise BackendError(
                    "SGLang runtime exited before becoming ready"
                    + (f":\n{log_tail}" if log_tail else "")
                )
            time.sleep(0.5)

        logs = self.process_manager.logs(lines=40)["lines"]
        log_tail = "\n".join(logs)
        raise BackendError(
            f"SGLang runtime did not become ready within {self.config.startup_timeout_seconds} seconds"
            + (f":\n{log_tail}" if log_tail else "")
        )

    def collect_metrics(self) -> RuntimeMetrics:
        metrics = super().collect_metrics()
        details: Dict[str, Any] = {"diagnostics": self.diagnostics().to_dict()}

        server_info = self._server_info()
        if server_info:
            details["server_info"] = server_info

        try:
            details["hicache_storage"] = self.hicache_storage_status()
        except BackendError as exc:
            details["hicache_storage_error"] = str(exc)

        cache_report = self.cache_report()
        if cache_report:
            details["cache_report"] = cache_report

        metrics_excerpt = self._metrics_excerpt()
        if metrics_excerpt:
            details["metrics_excerpt"] = metrics_excerpt

        metrics.details = details
        return metrics

    def start_runtime(self) -> Dict[str, Any]:
        diagnostics = self.diagnostics()
        if (
            self.config.enable_hierarchical_cache
            and diagnostics.hicache_supported is False
        ):
            raise BackendError(
                f"HiCache is not supported for {self.config.model_repo_id}: "
                f"{diagnostics.hicache_blocker}. Disable hierarchical cache for this model."
            )
        command = self._build_launch_command()
        env = self._build_env()
        result = self.process_manager.start(command, env=env)
        if result.get("reason") == "already_running":
            return {
                **result,
                "mode": "sglang",
                "command": self.process_manager.command(),
            }
        self._wait_until_ready()
        return {**result, "mode": "sglang"}

    def stop_runtime(self) -> Dict[str, Any]:
        return {**self.process_manager.stop(), "mode": "sglang"}

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        return {**self.process_manager.logs(lines=lines), "mode": "sglang"}

    def hicache_storage_status(self) -> Dict[str, Any]:
        response = self._request_admin("GET", "hicache/storage-backend", timeout=10)
        if not response.ok:
            raise BackendError(
                f"SGLang hicache status failed ({response.status_code}): {response.text.strip()}"
            )
        return _coerce_response_payload(response)

    def attach_hicache_storage_backend(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "hicache_storage_backend": self.config.hicache_storage_backend,
            "hicache_storage_prefetch_policy": self.config.hicache_storage_prefetch_policy,
            "hicache_write_policy": self.config.hicache_write_policy,
        }
        extra_config = self._build_hicache_extra_config()
        if extra_config:
            payload["hicache_storage_backend_extra_config_json"] = json.dumps(
                extra_config, sort_keys=True
            )

        if overrides:
            payload.update(overrides)
            raw_extra_config = payload.pop("hicache_storage_backend_extra_config", None)
            if raw_extra_config is not None:
                payload["hicache_storage_backend_extra_config_json"] = json.dumps(
                    raw_extra_config, sort_keys=True
                )

        response = self._request_admin(
            "PUT",
            "hicache/storage-backend",
            json=payload,
            timeout=30,
        )
        if not response.ok:
            raise BackendError(
                f"SGLang hicache attach failed ({response.status_code}): {response.text.strip()}"
            )
        return _coerce_response_payload(response)

    def detach_hicache_storage_backend(self) -> Dict[str, Any]:
        response = self._request_admin("DELETE", "hicache/storage-backend", timeout=30)
        if not response.ok:
            raise BackendError(
                f"SGLang hicache detach failed ({response.status_code}): {response.text.strip()}"
            )
        return _coerce_response_payload(response)

    def cache_report(self) -> Dict[str, Any]:
        server_info = self._server_info()
        if not server_info:
            return {}
        return {
            "enable_cache_report": bool(server_info.get("enable_cache_report")),
            "internal_states": server_info.get("internal_states", []),
            "page_size": server_info.get("page_size"),
            "enable_hierarchical_cache": server_info.get(
                "enable_hierarchical_cache"
            ),
            "hicache_storage_backend": server_info.get("hicache_storage_backend"),
        }
