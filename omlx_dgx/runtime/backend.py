# SPDX-License-Identifier: Apache-2.0
"""HTTP-backed runtime adapter for a TensorRT-LLM-compatible service."""

from __future__ import annotations

import csv
import io
import json
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests


class BackendError(RuntimeError):
    """Raised when the configured backend cannot be reached or proxying fails."""


def _parse_optional_number(value: Any) -> Optional[float]:
    text = str(value).strip()
    if not text or text in {"[N/A]", "N/A"}:
        return None
    if text.endswith("%"):
        text = text[:-1].strip()
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_int(value: Any) -> Optional[int]:
    number = _parse_optional_number(value)
    if number is None:
        return None
    return int(round(number))


def _parse_optional_bytes_to_mb(value: Any) -> Optional[int]:
    number = _parse_optional_number(value)
    if number is None:
        return None
    return int(number / (1024 * 1024))


def _read_system_memory_kb() -> Dict[str, int]:
    keys = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: Dict[str, int] = {}
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return values
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        name, _, raw_value = line.partition(":")
        if name not in keys:
            continue
        try:
            values[name] = int(raw_value.strip().split()[0])
        except (IndexError, ValueError):
            continue
    return values


def _extract_first_json_object(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    if start < 0:
        raise ValueError("no JSON object found in command output")
    return json.loads(raw_text[start:])


def _apply_nvidia_smi_metrics(metrics: RuntimeMetrics) -> None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = list(csv.reader(io.StringIO(result.stdout)))
    if not rows:
        raise RuntimeError("nvidia-smi returned no GPU rows")
    row = [item.strip() for item in rows[0]]
    metrics.gpu_name = row[0]
    metrics.gpu_memory_used_mb = _parse_optional_int(row[1])
    metrics.gpu_memory_total_mb = _parse_optional_int(row[2])
    metrics.gpu_util_percent = _parse_optional_int(row[3])
    metrics.gpu_temperature_c = _parse_optional_int(row[4])


def _apply_rocm_smi_metrics(metrics: RuntimeMetrics) -> Dict[str, Any]:
    result = subprocess.run(
        [
            "rocm-smi",
            "--showproductname",
            "--showuse",
            "--showtemp",
            "--showmeminfo",
            "vram",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = _extract_first_json_object(result.stdout)
    if not payload:
        raise RuntimeError("rocm-smi returned no GPU rows")
    card_key = sorted(payload.keys())[0]
    card = payload.get(card_key) or {}
    if not isinstance(card, dict):
        raise RuntimeError("rocm-smi returned an invalid JSON payload")

    metrics.gpu_name = (
        str(card.get("Card Series") or "").strip()
        or str(card.get("Card Model") or "").strip()
        or str(card.get("GFX Version") or "").strip()
        or None
    )
    metrics.gpu_memory_used_mb = _parse_optional_bytes_to_mb(
        card.get("VRAM Total Used Memory (B)")
    )
    metrics.gpu_memory_total_mb = _parse_optional_bytes_to_mb(
        card.get("VRAM Total Memory (B)")
    )
    metrics.gpu_util_percent = _parse_optional_int(card.get("GPU use (%)"))
    metrics.gpu_temperature_c = _parse_optional_int(
        card.get("Temperature (Sensor edge) (C)")
    )
    return {
        "gpu_metrics_card": card_key,
        "gpu_metrics_gfx_version": card.get("GFX Version"),
    }


@dataclass
class RuntimeMetrics:
    backend_url: str
    healthy: bool
    gpu_name: Optional[str] = None
    gpu_memory_used_mb: Optional[int] = None
    gpu_memory_total_mb: Optional[int] = None
    gpu_util_percent: Optional[int] = None
    gpu_temperature_c: Optional[int] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendCapabilities:
    chat_completions: bool = False
    completions: bool = False
    embeddings: bool = False
    rerank: bool = False
    vision_chat: bool = False
    ocr: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


class BackendAdapter(ABC):
    """Backend contract used by the DGX control-plane."""

    @abstractmethod
    def health(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def proxy(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        raise NotImplementedError

    @abstractmethod
    def collect_metrics(self) -> RuntimeMetrics:
        raise NotImplementedError

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    def start_runtime(self) -> Dict[str, Any]:
        raise BackendError("runtime start is not supported by this adapter")

    def stop_runtime(self) -> Dict[str, Any]:
        raise BackendError("runtime stop is not supported by this adapter")

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        raise BackendError("runtime logs are not supported by this adapter")

    def hicache_storage_status(self) -> Dict[str, Any]:
        raise BackendError("hicache storage status is not supported by this adapter")

    def attach_hicache_storage_backend(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        raise BackendError("hicache storage attach is not supported by this adapter")

    def detach_hicache_storage_backend(self) -> Dict[str, Any]:
        raise BackendError("hicache storage detach is not supported by this adapter")

    def cache_report(self) -> Dict[str, Any]:
        raise BackendError("cache report is not supported by this adapter")


class HttpOpenAIBackendAdapter(BackendAdapter):
    """Simple HTTP adapter for an OpenAI-compatible backend service."""

    def __init__(self, base_url: str, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        timeout = kwargs.pop("timeout", self.timeout)
        response = self.session.request(method, url, timeout=timeout, **kwargs)
        return response

    def health(self) -> bool:
        for path in ("health", "v1/models"):
            try:
                response = self._request("GET", path)
                if response.ok:
                    return True
            except requests.RequestException:
                continue
        return False

    def list_models(self) -> dict:
        response = self._request("GET", "v1/models")
        response.raise_for_status()
        return response.json()

    def proxy(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            response = self._request(method, path, **kwargs)
            return response
        except requests.RequestException as exc:
            raise BackendError(str(exc)) from exc

    def collect_metrics(self) -> RuntimeMetrics:
        metrics = RuntimeMetrics(backend_url=self.base_url, healthy=self.health())
        telemetry: Dict[str, Any] = {
            "gpu_metrics_source": None,
            "gpu_metrics_ok": False,
            "gpu_metrics_error": None,
            "system_memory_kb": _read_system_memory_kb(),
        }
        probe_errors = []
        probes = []
        if shutil.which("nvidia-smi"):
            probes.append(("nvidia-smi", _apply_nvidia_smi_metrics))
        if shutil.which("rocm-smi"):
            probes.append(("rocm-smi", _apply_rocm_smi_metrics))
        if not probes:
            telemetry["gpu_metrics_error"] = "no supported GPU telemetry CLI found"
        for source_name, probe in probes:
            try:
                telemetry["gpu_metrics_source"] = source_name
                extra = probe(metrics)
                telemetry["gpu_metrics_ok"] = True
                telemetry["gpu_metrics_error"] = None
                if isinstance(extra, dict):
                    telemetry.update(extra)
                break
            except FileNotFoundError as exc:
                probe_errors.append(f"{source_name}: {exc}")
            except subprocess.CalledProcessError as exc:
                probe_errors.append(
                    f"{source_name}: {exc.stderr.strip() or exc.stdout.strip() or str(exc)}"
                )
            except Exception as exc:
                probe_errors.append(f"{source_name}: {exc}")
        if not telemetry["gpu_metrics_ok"] and probe_errors:
            telemetry["gpu_metrics_error"] = "; ".join(probe_errors)
        metrics.details = {"telemetry": telemetry}
        return metrics

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            chat_completions=True,
            completions=True,
            embeddings=True,
            rerank=True,
        )


class ExternalOpenAIModelAdapter(HttpOpenAIBackendAdapter):
    """Static-capability adapter for externally managed OpenAI-compatible models."""

    def __init__(
        self,
        *,
        base_url: str,
        target_model_name: str,
        capabilities: BackendCapabilities,
    ) -> None:
        super().__init__(base_url)
        self.target_model_name = target_model_name
        self._capabilities = capabilities

    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def start_runtime(self) -> Dict[str, Any]:
        return {
            "started": self.health(),
            "mode": "openai_compatible_external",
            "target_model_name": self.target_model_name,
            "managed": False,
        }

    def stop_runtime(self) -> Dict[str, Any]:
        return {
            "stopped": False,
            "mode": "openai_compatible_external",
            "target_model_name": self.target_model_name,
            "managed": False,
        }

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        return {
            "lines": [],
            "path": None,
            "mode": "openai_compatible_external",
            "target_model_name": self.target_model_name,
            "managed": False,
        }
