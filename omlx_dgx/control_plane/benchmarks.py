# SPDX-License-Identifier: Apache-2.0
"""Stable benchmark report storage and execution helpers for DGX admin APIs."""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


class BenchmarkExecutionError(RuntimeError):
    """Raised when a benchmark run fails."""


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    script_path: str
    description: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


_BENCHMARK_SPECS: Dict[str, BenchmarkSpec] = {
    "qwen35-4b": BenchmarkSpec(
        name="qwen35-4b",
        script_path="scripts/bench_qwen35_4b.py",
        description="DGX chat benchmark for the managed Qwen3.5-4B control-plane path.",
    ),
    "multimodal-smoke": BenchmarkSpec(
        name="multimodal-smoke",
        script_path="scripts/bench_multimodal_smoke.py",
        description="DGX multimodal routing smoke benchmark for image chat and OCR capability gating.",
    ),
}

_CLI_OPTION_TYPES: Dict[str, str] = {
    "control_plane_url": "str",
    "runtime_url": "str",
    "model": "str",
    "long_prefix_repeat": "int",
    "long_output_max_tokens": "int",
    "target_context_tokens": "int",
    "prefix_salt": "str",
    "disable_thinking": "bool",
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _report_summary(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "report_id": payload.get("report_id"),
        "benchmark": payload.get("benchmark"),
        "created_at": payload.get("created_at"),
        "duration_sec": payload.get("duration_sec"),
        "control_plane_url": (
            payload.get("report", {})
            .get("urls", {})
            .get("control_plane_url")
        ),
        "runtime_url": (
            payload.get("report", {})
            .get("urls", {})
            .get("runtime_url")
        ),
    }


class BenchmarkReportStore:
    def __init__(self, root_path: str | Path) -> None:
        self.root_path = Path(root_path).expanduser().resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _benchmark_dir(self, benchmark_name: str) -> Path:
        path = self.root_path / benchmark_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_report(
        self,
        benchmark_name: str,
        *,
        command: Iterable[str],
        request_overrides: Dict[str, Any],
        report: Dict[str, Any],
        duration_sec: float,
    ) -> Dict[str, Any]:
        report_id = f"bench-{uuid.uuid4().hex[:12]}"
        payload = {
            "report_id": report_id,
            "benchmark": benchmark_name,
            "created_at": _utc_now(),
            "duration_sec": round(duration_sec, 3),
            "command": list(command),
            "request_overrides": dict(request_overrides),
            "report": report,
        }
        benchmark_dir = self._benchmark_dir(benchmark_name)
        report_path = benchmark_dir / f"{report_id}.json"
        latest_path = benchmark_dir / "latest.json"
        report_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        latest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return payload

    def latest_report(self, benchmark_name: str) -> Optional[Dict[str, Any]]:
        latest_path = self._benchmark_dir(benchmark_name) / "latest.json"
        if latest_path.exists():
            return json.loads(latest_path.read_text(encoding="utf-8"))

        candidates = sorted(
            (
                path for path in self._benchmark_dir(benchmark_name).glob("bench-*.json")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            return None
        return json.loads(candidates[0].read_text(encoding="utf-8"))

    def list_reports(self, benchmark_name: str, *, limit: int = 10) -> list[Dict[str, Any]]:
        reports = sorted(
            (
                path for path in self._benchmark_dir(benchmark_name).glob("bench-*.json")
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return [
            _report_summary(json.loads(path.read_text(encoding="utf-8")))
            for path in reports[: max(1, limit)]
        ]


class BenchmarkManager:
    def __init__(self, state_dir: str | Path) -> None:
        self.repo_root = Path(__file__).resolve().parents[2]
        self.store = BenchmarkReportStore(Path(state_dir) / "benchmarks")

    def available_benchmarks(self) -> list[Dict[str, Any]]:
        entries = []
        for spec in _BENCHMARK_SPECS.values():
            latest = self.store.latest_report(spec.name)
            entries.append(
                {
                    **spec.to_dict(),
                    "latest_report": None if latest is None else _report_summary(latest),
                }
            )
        return entries

    def latest_report(self, benchmark_name: str) -> Optional[Dict[str, Any]]:
        self._require_spec(benchmark_name)
        return self.store.latest_report(benchmark_name)

    def list_reports(self, benchmark_name: str, *, limit: int = 10) -> list[Dict[str, Any]]:
        self._require_spec(benchmark_name)
        return self.store.list_reports(benchmark_name, limit=limit)

    def run_benchmark(
        self,
        benchmark_name: str,
        *,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        spec = self._require_spec(benchmark_name)
        request_overrides = self._normalized_overrides(overrides or {})
        command = [
            sys.executable,
            str(self.repo_root / spec.script_path),
            *self._cli_args(request_overrides),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        duration_sec = time.perf_counter() - started
        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            details = stderr or stdout or f"exit code {completed.returncode}"
            raise BenchmarkExecutionError(
                f"benchmark '{benchmark_name}' failed: {details}"
            )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise BenchmarkExecutionError(
                f"benchmark '{benchmark_name}' returned invalid JSON: {exc}"
            ) from exc
        return self.store.write_report(
            benchmark_name,
            command=command,
            request_overrides=request_overrides,
            report=report,
            duration_sec=duration_sec,
        )

    def _require_spec(self, benchmark_name: str) -> BenchmarkSpec:
        spec = _BENCHMARK_SPECS.get(benchmark_name)
        if spec is None:
            raise KeyError(f"unknown benchmark: {benchmark_name}")
        return spec

    def _normalized_overrides(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in overrides.items():
            value_type = _CLI_OPTION_TYPES.get(key)
            if value_type is None or value is None:
                continue
            if value_type == "bool":
                normalized[key] = bool(value)
            elif value_type == "int":
                normalized[key] = int(value)
            else:
                normalized[key] = str(value)
        return normalized

    def _cli_args(self, overrides: Dict[str, Any]) -> list[str]:
        args: list[str] = []
        for key, value in overrides.items():
            flag = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    args.append(flag)
                continue
            args.extend([flag, str(value)])
        return args
