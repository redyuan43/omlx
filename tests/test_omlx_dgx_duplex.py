# SPDX-License-Identifier: Apache-2.0

import math
import struct
import time
from pathlib import Path

from omlx_dgx.duplex.audio_io import (
    EnergyVad,
    convert_wav_bytes,
    decode_wav_bytes,
    encode_pcm16_mono_wav,
)
from omlx_dgx.duplex.config import DuplexConfig
from omlx_dgx.duplex.session import DuplexSession, default_playback_sink


def _sine_wav(sample_rate: int, *, seconds: float, channels: int = 1) -> bytes:
    frames = []
    total_frames = int(sample_rate * seconds)
    for index in range(total_frames):
        sample = int(12000 * math.sin(2 * math.pi * 440.0 * (index / sample_rate)))
        packed = struct.pack("<h", sample)
        if channels == 2:
            packed = packed * 2
        frames.append(packed)
    raw = b"".join(frames)
    return encode_pcm16_mono_wav(raw, sample_rate=sample_rate) if channels == 1 else _encode_stereo(raw, sample_rate)


def _encode_stereo(frames: bytes, sample_rate: int) -> bytes:
    from omlx_dgx.duplex.audio_io import WavePayload, encode_wav_bytes

    return encode_wav_bytes(
        WavePayload(
            sample_rate=sample_rate,
            channels=2,
            sample_width=2,
            frames=frames,
        )
    )


class FakeAsrClient:
    def __init__(self, texts):
        self.texts = list(texts)

    def transcribe_wav_bytes(self, wav_bytes: bytes, *, trace_id: str = ""):
        text = self.texts.pop(0)
        return type(
            "AsrResult",
            (),
            {
                "text": text,
                "latency_ms": 12,
                "duration_s": 1.0,
                "language": "zh",
                "confidence": 0.9,
                "raw": {"trace_id": trace_id, "text": text},
            },
        )()


class FakeTtsClient:
    def __init__(self):
        self.calls = []

    def speak(self, text: str, **kwargs):
        self.calls.append((text, kwargs))
        wav_bytes = _sine_wav(24000, seconds=0.8)
        return type(
            "TtsResult",
            (),
            {
                "wav_bytes": wav_bytes,
                "latency_ms": 23,
                "duration_ms": 800.0,
                "worker_id": "0",
            },
        )()


class FakeLlmClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return type(
            "LlmResult",
            (),
            {
                "text": self.replies.pop(0),
                "latency_ms": 34,
                "raw": {"choices": [{"message": {"content": self.replies[0] if self.replies else ""}}]},
            },
        )()


def test_convert_wav_bytes_resamples_to_16k_mono():
    stereo_wav = _sine_wav(24000, seconds=0.2, channels=2)
    converted = convert_wav_bytes(
        stereo_wav,
        target_sample_rate=16000,
        target_channels=1,
        target_sample_width=2,
    )
    payload = decode_wav_bytes(converted)
    assert payload.sample_rate == 16000
    assert payload.channels == 1
    assert payload.sample_width == 2
    assert payload.duration_ms > 150


def test_energy_vad_detects_speech_segment():
    vad = EnergyVad(
        threshold=0.02,
        frame_ms=20,
        start_ms=60,
        stop_ms=80,
        min_utterance_ms=120,
        preroll_ms=40,
    )
    silence = b"\x00\x00" * 320
    speech = struct.pack("<h", 10000) * 320
    utterance = None
    for _ in range(3):
        assert vad.process(silence) is None
    for _ in range(8):
        utterance = vad.process(speech)
    for _ in range(6):
        utterance = vad.process(silence)
        if utterance:
            break
    assert utterance is not None
    assert len(utterance) > 0


def test_duplex_session_tracks_turns_and_barge_in(tmp_path: Path):
    config = DuplexConfig(runtime_dir=str(tmp_path))
    config.services.model = "qwen35-4b"
    session = DuplexSession(
        config=config,
        asr_client=FakeAsrClient(["你好", "停一下"]),
        tts_client=FakeTtsClient(),
        llm_client=FakeLlmClient(["你好，我在这里。", "好的，我停下。"]),
        playback_sink=default_playback_sink(simulated=True),
    )

    first = session.submit_user_audio(_sine_wav(16000, seconds=0.3), prompt_label="greeting")
    assert session.is_playing() is True
    interrupted = session.interrupt_playback(reason="test")
    assert interrupted is True
    second = session.submit_user_audio(_sine_wav(16000, seconds=0.3), prompt_label="interrupt")
    assert second.result.turn_index == 2
    assert session.interrupt_count == 1
    assert first.result.interrupted is True
    assert second.result.transcript == "停一下"
    assert session.wait_for_playback(timeout=2.0) is True
    assert session.state == "listening"
    assert len(session.turns) == 2
    assert session.turns[0].user_audio_path.endswith("user.wav")
    assert Path(session.turns[0].reply_audio_path).exists()
    assert session.turns[1].metadata["conversation_id"] == session.conversation_id
