# SPDX-License-Identifier: Apache-2.0
"""Client for the local CapsWriter ASR HTTP service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass(frozen=True)
class AsrResult:
    text: str
    latency_ms: int
    duration_s: float
    language: str
    confidence: float
    raw: Dict[str, Any]


class AsrClient:
    def __init__(self, base_url: str, *, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def health(self) -> Dict[str, Any]:
        response = self.session.get(
            f"{self.base_url}/api/health",
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def transcribe_wav_bytes(self, wav_bytes: bytes, *, trace_id: str = "") -> AsrResult:
        request_started = time.perf_counter()
        response = self.session.post(
            f"{self.base_url}/api/asr/transcribe",
            files={"audio": ("input.wav", wav_bytes, "audio/wav")},
            data={"trace_id": trace_id} if trace_id else None,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = int((time.perf_counter() - request_started) * 1000)
        return AsrResult(
            text=str(payload.get("text", "") or "").strip(),
            latency_ms=latency_ms,
            duration_s=float(payload.get("duration", 0.0) or 0.0),
            language=str(payload.get("language", "") or ""),
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            raw=payload,
        )
