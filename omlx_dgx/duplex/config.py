# SPDX-License-Identifier: Apache-2.0
"""Configuration models for the experimental duplex pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DuplexAudioConfig:
    capture_sample_rate: int = 16000
    capture_channels: int = 1
    frame_ms: int = 20
    vad_threshold: float = 0.018
    vad_start_ms: int = 160
    vad_stop_ms: int = 400
    interrupt_vad_threshold: float = 0.016
    min_utterance_ms: int = 240
    preroll_ms: int = 120
    capture_device: str = ""
    playback_device: str = ""


@dataclass
class DuplexServiceConfig:
    llm_base_url: str = "http://127.0.0.1:8008"
    model: str = "qwen35-4b"
    asr_url: str = "http://127.0.0.1:8001"
    tts_url: str = "http://127.0.0.1:8002"
    speaker: str = ""
    system_prompt: str = (
        "You are a low-latency bilingual voice assistant. "
        "Keep replies concise and conversational."
    )
    tts_speed: float = 1.0
    request_timeout_s: float = 180.0
    stream_llm: bool = True
    segment_tts: bool = True
    stream_sentence_min_chars: int = 12
    stream_sentence_max_chars: int = 80


@dataclass
class DuplexSelfTestThresholds:
    prompt_similarity: float = 0.75
    assistant_similarity: float = 0.30
    tail_similarity: float = 0.30
    min_barge_in_count: int = 1


@dataclass
class DuplexConfig:
    runtime_dir: str = ".runtime/duplex-cli"
    session_id: Optional[str] = None
    audio: DuplexAudioConfig = field(default_factory=DuplexAudioConfig)
    services: DuplexServiceConfig = field(default_factory=DuplexServiceConfig)
    selftest: DuplexSelfTestThresholds = field(default_factory=DuplexSelfTestThresholds)
    selftest_play_user_prompts: bool = True

    def runtime_path(self) -> Path:
        return Path(self.runtime_dir).expanduser()
