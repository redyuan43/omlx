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
    SequentialPlaybackHandle,
    SimulatedPlaybackSink,
    concatenate_wav_bytes,
    convert_wav_bytes,
    encode_pcm16_mono_wav,
    detect_default_playback_target,
)
from .config import DuplexConfig
from .llm_client import LLMChatClient
from .tts_client import TtsClient


EventCallback = Callable[[str, Dict[str, Any]], None]

_SENTENCE_BREAKS = "。！？!?；;\n"


def _drain_ready_sentences(
    buffer: str,
    *,
    min_chars: int,
    max_chars: int,
) -> List[str]:
    ready: List[str] = []
    consumed = 0
    for index, char in enumerate(buffer):
        current = buffer[consumed : index + 1]
        if char in _SENTENCE_BREAKS and len(current.strip()) >= min_chars:
            ready.append(current.strip())
            consumed = index + 1
            continue
        if len(current.strip()) >= max_chars and ("，" in current or "," in current or "、" in current):
            split_at = max(current.rfind("，"), current.rfind(","), current.rfind("、"))
            if split_at >= 0:
                candidate = current[: split_at + 1].strip()
                if candidate:
                    ready.append(candidate)
                    consumed += split_at + 1
    return ready


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
    done: Optional[threading.Event] = None

    def wait_complete(self, timeout: Optional[float] = None) -> bool:
        if self.done is None:
            return True
        return self.done.wait(timeout)


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
        reply_audio_path = turn_dir / "assistant.wav"
        result = DuplexTurnResult(
            turn_index=turn_index,
            prompt_label=prompt_label,
            transcript=asr_result.text,
            reply_text="",
            asr_ms=asr_result.latency_ms,
            llm_ms=0,
            tts_ms=0,
            total_ms=0,
            speak_latency_ms=0,
            interrupted=False,
            user_audio_path=str(user_audio_path),
            reply_audio_path=str(reply_audio_path),
            metadata={
                "conversation_id": self.conversation_id,
                "asr": asr_result.raw,
                "llm": {},
                "tts_duration_ms": 0.0,
                "tts_worker_id": "",
            },
        )
        playback = SequentialPlaybackHandle(self.playback_sink)
        done = threading.Event()
        with self._lock:
            self.turns.append(result)
            self._current_turn = result
            self._playback = playback
            self._state = "thinking"
        worker = threading.Thread(
            target=self._generate_and_speak_turn,
            args=(
                turn_index,
                prompt_label,
                request_started,
                result,
                playback,
                done,
            ),
            daemon=True,
        )
        worker.start()
        watcher = threading.Thread(target=self._watch_playback, args=(playback,), daemon=True)
        watcher.start()
        return DuplexTurnContext(result=result, playback=playback, done=done)

    def _generate_and_speak_turn(
        self,
        turn_index: int,
        prompt_label: str,
        request_started: float,
        result: DuplexTurnResult,
        playback: SequentialPlaybackHandle,
        done: threading.Event,
    ) -> None:
        wav_chunks: List[bytes] = []
        llm_started = time.perf_counter()
        llm_raw: Dict[str, Any] = {}
        full_text = ""
        first_audio_started = False
        tts_latency_total = 0
        tts_duration_total = 0.0
        tts_worker_id = ""
        buffer = ""
        try:
            messages_snapshot = list(self._messages)
            if self.config.services.stream_llm:
                for delta in self.llm_client.chat_stream(
                    model=self.config.services.model,
                    messages=messages_snapshot,
                    conversation_id=self.conversation_id,
                    system_prompt=self.config.services.system_prompt,
                ):
                    full_text += delta
                    buffer += delta
                    if result.interrupted:
                        break
                    for sentence in _drain_ready_sentences(
                        buffer,
                        min_chars=self.config.services.stream_sentence_min_chars,
                        max_chars=self.config.services.stream_sentence_max_chars,
                    ):
                        buffer = buffer[len(sentence) :]
                        chunk_result = self.tts_client.speak(
                            sentence,
                            speaker=self.config.services.speaker,
                            speed=self.config.services.tts_speed,
                            trace_id=f"{self.session_id}-turn{turn_index}-tts",
                        )
                        tts_latency_total += chunk_result.latency_ms
                        tts_duration_total += chunk_result.duration_ms
                        tts_worker_id = chunk_result.worker_id or tts_worker_id
                        wav_chunks.append(chunk_result.wav_bytes)
                        playback.enqueue(chunk_result.wav_bytes)
                        if not first_audio_started:
                            first_audio_started = True
                            result.speak_latency_ms = int(
                                (time.perf_counter() - request_started) * 1000
                            )
                            with self._lock:
                                self._state = "speaking"
                            self._emit(
                                "assistant_playback_started",
                                {
                                    "turn_index": turn_index,
                                    "prompt_label": prompt_label,
                                    "reply_text": full_text.strip(),
                                    "reply_audio_path": result.reply_audio_path,
                                },
                            )
                if buffer.strip() and not result.interrupted:
                    chunk_result = self.tts_client.speak(
                        buffer.strip(),
                        speaker=self.config.services.speaker,
                        speed=self.config.services.tts_speed,
                        trace_id=f"{self.session_id}-turn{turn_index}-tts-final",
                    )
                    tts_latency_total += chunk_result.latency_ms
                    tts_duration_total += chunk_result.duration_ms
                    tts_worker_id = chunk_result.worker_id or tts_worker_id
                    wav_chunks.append(chunk_result.wav_bytes)
                    playback.enqueue(chunk_result.wav_bytes)
                    if not first_audio_started:
                        first_audio_started = True
                        result.speak_latency_ms = int(
                            (time.perf_counter() - request_started) * 1000
                        )
                        with self._lock:
                            self._state = "speaking"
                        self._emit(
                            "assistant_playback_started",
                            {
                                "turn_index": turn_index,
                                "prompt_label": prompt_label,
                                "reply_text": full_text.strip(),
                                "reply_audio_path": result.reply_audio_path,
                            },
                        )
                result.llm_ms = int((time.perf_counter() - llm_started) * 1000)
                llm_raw = {"streamed": True}
            else:
                llm_result = self.llm_client.chat(
                    model=self.config.services.model,
                    messages=messages_snapshot,
                    conversation_id=self.conversation_id,
                    system_prompt=self.config.services.system_prompt,
                )
                full_text = llm_result.text
                result.llm_ms = llm_result.latency_ms
                llm_raw = llm_result.raw
                chunk_result = self.tts_client.speak(
                    full_text,
                    speaker=self.config.services.speaker,
                    speed=self.config.services.tts_speed,
                    trace_id=f"{self.session_id}-turn{turn_index}-tts",
                )
                tts_latency_total += chunk_result.latency_ms
                tts_duration_total += chunk_result.duration_ms
                tts_worker_id = chunk_result.worker_id or tts_worker_id
                wav_chunks.append(chunk_result.wav_bytes)
                playback.enqueue(chunk_result.wav_bytes)
                result.speak_latency_ms = int((time.perf_counter() - request_started) * 1000)
                with self._lock:
                    self._state = "speaking"
                self._emit(
                    "assistant_playback_started",
                    {
                        "turn_index": turn_index,
                        "prompt_label": prompt_label,
                        "reply_text": full_text.strip(),
                        "reply_audio_path": result.reply_audio_path,
                    },
                )

            result.reply_text = full_text.strip()
            result.tts_ms = tts_latency_total
            result.total_ms = int((time.perf_counter() - request_started) * 1000)
            result.metadata["llm"] = llm_raw
            result.metadata["tts_duration_ms"] = tts_duration_total
            result.metadata["tts_worker_id"] = tts_worker_id
            with self._lock:
                self._messages.append({"role": "assistant", "content": result.reply_text})
            if wav_chunks:
                Path(result.reply_audio_path).write_bytes(concatenate_wav_bytes(wav_chunks))
            self._write_turn_metadata(result)
        finally:
            playback.close_input()
            done.set()

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
        threading.Thread(
            target=self._announce_completion,
            args=(context,),
            daemon=True,
        ).start()

    def _announce_completion(self, context: DuplexTurnContext) -> None:
        context.wait_complete(timeout=max(30.0, self.config.services.request_timeout_s))
        result = context.result
        print(f"assistant> {result.reply_text}")
        print(
            f"[speaking] asr={result.asr_ms}ms llm={result.llm_ms}ms "
            f"tts={result.tts_ms}ms first_audio={result.speak_latency_ms}ms "
            f"total={result.total_ms}ms"
        )


def default_playback_sink(*, simulated: bool, device: str = "") -> Any:
    if simulated:
        return SimulatedPlaybackSink()
    return AplayPlaybackSink(device=device)
