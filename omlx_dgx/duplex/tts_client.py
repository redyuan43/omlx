# SPDX-License-Identifier: Apache-2.0
"""Client for the local CapsWriter TTS HTTP service."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .audio_io import decode_wav_bytes


@dataclass(frozen=True)
class TtsResult:
    wav_bytes: bytes
    latency_ms: int
    duration_ms: float
    worker_id: str


class TtsClient:
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

    def ensure_loaded(self, *, timeout_s: float = 180.0) -> Dict[str, Any]:
        health = self.health()
        if health.get("tts_model_loaded"):
            return health
        if health.get("status") != "loading":
            response = self.session.post(
                f"{self.base_url}/api/tts/load",
                timeout=self.timeout,
            )
            response.raise_for_status()
        deadline = time.monotonic() + timeout_s
        last_payload: Dict[str, Any] = health
        while time.monotonic() < deadline:
            time.sleep(0.5)
            last_payload = self.health()
            if last_payload.get("tts_model_loaded"):
                return last_payload
            if last_payload.get("status") not in {"loading", "idle", "healthy"}:
                break
        raise RuntimeError(
            "TTS model did not become ready: "
            + str(last_payload.get("tts_init_error") or last_payload)
        )

    def speak(
        self,
        text: str,
        *,
        speaker: str = "",
        speed: float = 1.0,
        instruction: str = "",
        trace_id: str = "",
    ) -> TtsResult:
        self.ensure_loaded(timeout_s=self.timeout)
        request_started = time.perf_counter()
        response = self.session.post(
            f"{self.base_url}/api/tts/speak",
            json={
                "text": text,
                "speaker": speaker or None,
                "speed": speed,
                "instruction": instruction or None,
                "trace_id": trace_id or None,
            },
            timeout=self.timeout,
        )
        if response.status_code == 503:
            self.ensure_loaded(timeout_s=self.timeout)
            response = self.session.post(
                f"{self.base_url}/api/tts/speak",
                json={
                    "text": text,
                    "speaker": speaker or None,
                    "speed": speed,
                    "instruction": instruction or None,
                    "trace_id": trace_id or None,
                },
                timeout=self.timeout,
            )
        response.raise_for_status()
        wav_bytes = response.content
        payload = decode_wav_bytes(wav_bytes)
        return TtsResult(
            wav_bytes=wav_bytes,
            latency_ms=int((time.perf_counter() - request_started) * 1000),
            duration_ms=payload.duration_ms,
            worker_id=str(response.headers.get("X-TTS-Worker-Id", "")),
        )

    def plan(self, text: str, *, trace_id: str = "") -> Dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/tts/plan",
            json={"text": text, "trace_id": trace_id or None},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
