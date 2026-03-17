#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Local MiniCPM-o-style duplex CLI backed by ASR + Qwen3.5 + TTS."""

from __future__ import annotations

import argparse

from omlx_dgx.duplex.asr_client import AsrClient
from omlx_dgx.duplex.config import DuplexConfig
from omlx_dgx.duplex.llm_client import LLMChatClient
from omlx_dgx.duplex.session import DuplexLiveRunner, DuplexSession, default_playback_sink
from omlx_dgx.duplex.tts_client import TtsClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the experimental duplex CLI")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", default="qwen35-4b")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8001")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8002")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--record-dir", default=".runtime/duplex-cli")
    parser.add_argument("--capture-device", default="")
    parser.add_argument("--playback-device", default="")
    parser.add_argument("--vad-threshold", type=float, default=0.018)
    parser.add_argument("--vad-start-ms", type=int, default=160)
    parser.add_argument("--vad-stop-ms", type=int, default=400)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = DuplexConfig(runtime_dir=args.record_dir)
    config.services.llm_base_url = args.llm_base_url
    config.services.model = args.model
    config.services.asr_url = args.asr_url
    config.services.tts_url = args.tts_url
    config.services.speaker = args.speaker
    if args.system_prompt:
        config.services.system_prompt = args.system_prompt
    config.audio.capture_device = args.capture_device
    config.audio.playback_device = args.playback_device
    config.audio.vad_threshold = args.vad_threshold
    config.audio.interrupt_vad_threshold = args.vad_threshold * 0.9
    config.audio.vad_start_ms = args.vad_start_ms
    config.audio.vad_stop_ms = args.vad_stop_ms

    session = DuplexSession(
        config=config,
        asr_client=AsrClient(config.services.asr_url),
        tts_client=TtsClient(config.services.tts_url),
        llm_client=LLMChatClient(config.services.llm_base_url),
        playback_sink=default_playback_sink(
            simulated=False,
            device=config.audio.playback_device,
        ),
    )
    runner = DuplexLiveRunner(session, config=config)
    runner.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
