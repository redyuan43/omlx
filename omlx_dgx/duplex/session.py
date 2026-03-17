# SPDX-License-Identifier: Apache-2.0
"""Session orchestration for the experimental duplex pipeline."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .asr_client import AsrClient
from .audio_io import (
    ARecordCaptureStream,
    AplayPlaybackSink,
    EnergyVad,
    PlaybackHandle,
    SimulatedPlaybackSink,
    convert_wav_bytes,
    encode_pcm16_mono_wav,
)
from .config import DuplexConfig
from .llm_client import LLMChatClient
from .tts_client import TtsClient


EventCallback = Callable[[str, Dict[str, Any]], None]


@dataclass
class DuplexTurnResult:
    turn_index: int
    prompt_label: str
    transcript: str
    reply_text: str
    asr_ms: int
    llm_ms: int
    tts_ms: int
    total_ms: int
    speak_latency_ms: int
    interrupted: bool
    user_audio_path: str
    reply_audio_path: str
    assistant_transcript: str = ""
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DuplexTurnContext:
    result: DuplexTurnResult
    playback: Optional[PlaybackHandle]


class DuplexSession:
    """Turn-level duplex orchestrator shared by CLI and selftest."""

    def __init__(
        self,
        *,
        config: DuplexConfig,
        asr_client: AsrClient,
        tts_client: TtsClient,
        llm_client: LLMChatClient,
        playback_sink: Any,
        event_callback: Optional[EventCallback] = None,
    ) -> None:
        self.config = config
        self.asr_client = asr_client
        self.tts_client = tts_client
        self.llm_client = llm_client
        self.playback_sink = playback_sink
        self.event_callback = event_callback

        self.session_id = config.session_id or time.strftime("duplex-%Y%m%d-%H%M%S")
        self.conversation_id = f"{self.session_id}-{uuid.uuid4().hex[:8]}"
        self.runtime_dir = config.runtime_path() / self.session_id
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.turns: List[DuplexTurnResult] = []
        self._messages: List[Dict[str, Any]] = []
        if config.services.system_prompt:
            self._messages.append(
                {"role": "system", "content": config.services.system_prompt}
            )
        self._lock = threading.RLock()
        self._playback: Optional[PlaybackHandle] = None
        self._current_turn: Optional[DuplexTurnResult] = None
        self._state = "listening"
        self._interrupt_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def interrupt_count(self) -> int:
        return self._interrupt_count

    def is_playing(self) -> bool:
        with self._lock:
            return bool(self._playback and self._playback.is_active())

    def is_thinking(self) -> bool:
        with self._lock:
            return self._state == "thinking"

    def close(self) -> None:
        self.interrupt_playback(reason="session_close")

    def interrupt_playback(self, *, reason: str = "user_barge_in") -> bool:
        with self._lock:
            playback = self._playback
            current_turn = self._current_turn
            if playback is None or not playback.is_active():
                return False
            playback.stop()
            self._interrupt_count += 1
            self._state = "interrupted"
            if current_turn is not None:
                current_turn.interrupted = True
            self._emit(
                "assistant_interrupted",
                {
                    "reason": reason,
                    "interrupt_count": self._interrupt_count,
                    "turn_index": current_turn.turn_index if current_turn else None,
                },
            )
            return True

    def wait_for_playback(self, *, timeout: Optional[float] = None) -> bool:
        with self._lock:
            playback = self._playback
        if playback is None:
            return True
        finished = playback.wait(timeout)
        if finished:
            self._mark_playback_finished(playback)
        return finished

    def submit_user_audio(
        self,
        wav_bytes: bytes,
        *,
        prompt_label: str,
    ) -> DuplexTurnContext:
        request_started = time.perf_counter()
        with self._lock:
            turn_index = len(self.turns) + 1
            self._state = "thinking"
        turn_dir = self.runtime_dir / f"turn-{turn_index:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        user_audio_path = turn_dir / "user.wav"
        user_audio_path.write_bytes(wav_bytes)
        self._emit(
            "user_audio_submitted",
            {"turn_index": turn_index, "prompt_label": prompt_label},
        )

        asr_input = convert_wav_bytes(
            wav_bytes,
            target_sample_rate=self.config.audio.capture_sample_rate,
            target_channels=self.config.audio.capture_channels,
            target_sample_width=2,
        )
        asr_result = self.asr_client.transcribe_wav_bytes(
            asr_input,
            trace_id=f"{self.session_id}-turn{turn_index}-asr",
        )

        with self._lock:
            self._messages.append({"role": "user", "content": asr_result.text})

        llm_result = self.llm_client.chat(
            model=self.config.services.model,
            messages=list(self._messages),
            conversation_id=self.conversation_id,
            system_prompt=self.config.services.system_prompt,
        )
        with self._lock:
            self._messages.append({"role": "assistant", "content": llm_result.text})

        tts_result = self.tts_client.speak(
            llm_result.text,
            speaker=self.config.services.speaker,
            speed=self.config.services.tts_speed,
            trace_id=f"{self.session_id}-turn{turn_index}-tts",
        )
        reply_audio_path = turn_dir / "assistant.wav"
        reply_audio_path.write_bytes(tts_result.wav_bytes)
        speak_latency_ms = int((time.perf_counter() - request_started) * 1000)
        result = DuplexTurnResult(
            turn_index=turn_index,
            prompt_label=prompt_label,
            transcript=asr_result.text,
            reply_text=llm_result.text,
            asr_ms=asr_result.latency_ms,
            llm_ms=llm_result.latency_ms,
            tts_ms=tts_result.latency_ms,
            total_ms=speak_latency_ms,
            speak_latency_ms=speak_latency_ms,
            interrupted=False,
            user_audio_path=str(user_audio_path),
            reply_audio_path=str(reply_audio_path),
            metadata={
                "conversation_id": self.conversation_id,
                "asr": asr_result.raw,
                "llm": llm_result.raw,
                "tts_duration_ms": tts_result.duration_ms,
                "tts_worker_id": tts_result.worker_id,
            },
        )
        playback = self.playback_sink.play(tts_result.wav_bytes)
        with self._lock:
            self.turns.append(result)
            self._current_turn = result
            self._playback = playback
            self._state = "speaking"
        self._write_turn_metadata(result)
        self._emit(
            "assistant_playback_started",
            {
                "turn_index": turn_index,
                "prompt_label": prompt_label,
                "reply_text": llm_result.text,
                "reply_audio_path": str(reply_audio_path),
            },
        )
        watcher = threading.Thread(
            target=self._watch_playback,
            args=(playback,),
            daemon=True,
        )
        watcher.start()
        return DuplexTurnContext(result=result, playback=playback)

    def _watch_playback(self, playback: PlaybackHandle) -> None:
        playback.wait()
        self._mark_playback_finished(playback)

    def _mark_playback_finished(self, playback: PlaybackHandle) -> None:
        with self._lock:
            if self._playback is not playback:
                return
            finished_turn = self._current_turn
            self._playback = None
            self._current_turn = None
            self._state = "listening"
        if finished_turn is not None:
            self._emit(
                "assistant_playback_finished",
                {
                    "turn_index": finished_turn.turn_index,
                    "interrupted": finished_turn.interrupted,
                },
            )

    def _emit(self, event_name: str, payload: Dict[str, Any]) -> None:
        if self.event_callback is None:
            return
        self.event_callback(event_name, payload)

    def _write_turn_metadata(self, result: DuplexTurnResult) -> None:
        turn_dir = Path(result.user_audio_path).parent
        metadata_path = turn_dir / "turn.json"
        metadata_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class DuplexLiveRunner:
    """Realtime CLI loop using ALSA capture and the duplex session."""

    def __init__(self, session: DuplexSession, *, config: DuplexConfig) -> None:
        self.session = session
        self.config = config

    def run(self) -> None:
        capture = ARecordCaptureStream(
            sample_rate=self.config.audio.capture_sample_rate,
            channels=self.config.audio.capture_channels,
            frame_ms=self.config.audio.frame_ms,
            device=self.config.audio.capture_device,
        )
        listen_vad = EnergyVad(
            threshold=self.config.audio.vad_threshold,
            frame_ms=self.config.audio.frame_ms,
            start_ms=self.config.audio.vad_start_ms,
            stop_ms=self.config.audio.vad_stop_ms,
            min_utterance_ms=self.config.audio.min_utterance_ms,
            preroll_ms=self.config.audio.preroll_ms,
        )
        interrupt_vad = EnergyVad(
            threshold=self.config.audio.interrupt_vad_threshold,
            frame_ms=self.config.audio.frame_ms,
            start_ms=self.config.audio.vad_start_ms,
            stop_ms=self.config.audio.vad_stop_ms,
            min_utterance_ms=self.config.audio.min_utterance_ms,
            preroll_ms=self.config.audio.preroll_ms,
        )
        capture.start()
        print("[listening] speak to start; Ctrl+C to stop")
        try:
            while True:
                frame = capture.read_frame(timeout=0.1)
                if frame is None:
                    break
                if not frame:
                    continue
                if self.session.is_thinking():
                    continue
                if self.session.is_playing():
                    utterance = interrupt_vad.process(frame)
                    if interrupt_vad.active:
                        self.session.interrupt_playback()
                    if utterance:
                        self._submit_live_utterance(utterance, prompt_label="barge_in")
                        interrupt_vad.reset()
                        listen_vad.reset()
                    continue
                utterance = listen_vad.process(frame)
                if utterance:
                    self._submit_live_utterance(utterance, prompt_label="live_user")
                    listen_vad.reset()
                    interrupt_vad.reset()
        except KeyboardInterrupt:
            print("\n[stopped]")
        finally:
            capture.close()
            self.session.close()

    def _submit_live_utterance(self, pcm_bytes: bytes, *, prompt_label: str) -> None:
        wav_bytes = encode_pcm16_mono_wav(
            pcm_bytes,
            sample_rate=self.config.audio.capture_sample_rate,
        )
        print("[thinking]")
        context = self.session.submit_user_audio(wav_bytes, prompt_label=prompt_label)
        result = context.result
        print(f"user> {result.transcript}")
        print(f"assistant> {result.reply_text}")
        print(
            f"[speaking] asr={result.asr_ms}ms llm={result.llm_ms}ms "
            f"tts={result.tts_ms}ms total={result.total_ms}ms"
        )


def default_playback_sink(*, simulated: bool, device: str = "") -> Any:
    if simulated:
        return SimulatedPlaybackSink()
    return AplayPlaybackSink(device=device)
