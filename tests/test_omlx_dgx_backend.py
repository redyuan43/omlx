# SPDX-License-Identifier: Apache-2.0

import subprocess

import omlx_dgx.runtime.backend as backend_module
from omlx_dgx.runtime.backend import HttpOpenAIBackendAdapter


def test_http_openai_backend_adapter_uses_rocm_smi_when_available(monkeypatch):
    adapter = HttpOpenAIBackendAdapter("http://127.0.0.1:1234")
    monkeypatch.setattr(adapter, "health", lambda: True)

    def fake_which(binary: str):
        if binary == "nvidia-smi":
            return None
        if binary == "rocm-smi":
            return "/usr/bin/rocm-smi"
        return None

    def fake_run(args, **kwargs):
        assert args[0] == "rocm-smi"
        return subprocess.CompletedProcess(
            args,
            0,
            stdout=(
                "WARNING: AMD GPU device(s) is/are in a low-power state. Check power control/runtime_status\n"
                '{"card0": {"Temperature (Sensor edge) (C)": "41.0", "GPU use (%)": "7", '
                '"VRAM Total Memory (B)": "103079215104", "VRAM Total Used Memory (B)": "249303040", '
                '"Card Series": "AMD Radeon Graphics", "GFX Version": "gfx1151"}}\n'
            ),
            stderr="",
        )

    monkeypatch.setattr(backend_module.shutil, "which", fake_which)
    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)

    metrics = adapter.collect_metrics().to_dict()
    telemetry = metrics["details"]["telemetry"]

    assert metrics["healthy"] is True
    assert metrics["gpu_name"] == "AMD Radeon Graphics"
    assert metrics["gpu_util_percent"] == 7
    assert metrics["gpu_temperature_c"] == 41
    assert metrics["gpu_memory_total_mb"] == 98304
    assert metrics["gpu_memory_used_mb"] == 237
    assert telemetry["gpu_metrics_ok"] is True
    assert telemetry["gpu_metrics_source"] == "rocm-smi"
    assert telemetry["gpu_metrics_card"] == "card0"
    assert telemetry["gpu_metrics_gfx_version"] == "gfx1151"


def test_http_openai_backend_adapter_reports_probe_failures(monkeypatch):
    adapter = HttpOpenAIBackendAdapter("http://127.0.0.1:1234")
    monkeypatch.setattr(adapter, "health", lambda: False)

    def fake_which(binary: str):
        if binary in {"nvidia-smi", "rocm-smi"}:
            return f"/usr/bin/{binary}"
        return None

    def fake_run(args, **kwargs):
        if args[0] == "nvidia-smi":
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=args,
                stderr="Failed to initialize NVML: Unknown Error",
            )
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=args,
            stderr="ROCm SMI returned an unexpected payload",
        )

    monkeypatch.setattr(backend_module.shutil, "which", fake_which)
    monkeypatch.setattr(backend_module.subprocess, "run", fake_run)

    metrics = adapter.collect_metrics().to_dict()
    telemetry = metrics["details"]["telemetry"]

    assert telemetry["gpu_metrics_ok"] is False
    assert telemetry["gpu_metrics_source"] == "rocm-smi"
    assert "nvidia-smi" in telemetry["gpu_metrics_error"]
    assert "rocm-smi" in telemetry["gpu_metrics_error"]
    assert "MemTotal" in telemetry["system_memory_kb"]
