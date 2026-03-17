# SPDX-License-Identifier: Apache-2.0
"""Audio helpers for duplex capture, playback, and test injection."""

from __future__ import annotations

import audioop
import io
import queue
import subprocess
import tempfile
import threading
import time
import wave
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class WavePayload:
    sample_rate: int
    channels: int
    sample_width: int
    frames: bytes

    @property
    def duration_ms(self) -> float:
        if not self.frames or not self.sample_rate or not self.channels or not self.sample_width:
            return 0.0
        frame_size = self.channels * self.sample_width
        if frame_size <= 0:
            return 0.0
        total_frames = len(self.frames) / frame_size
        return (total_frames / float(self.sample_rate)) * 1000.0


def decode_wav_bytes(wav_bytes: bytes) -> WavePayload:
    with wave.open(io.BytesIO(wav_bytes), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    return WavePayload(
        sample_rate=sample_rate,
        channels=channels,
        sample_width=sample_width,
        frames=frames,
    )


def encode_wav_bytes(payload: WavePayload) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(payload.channels)
        handle.setsampwidth(payload.sample_width)
        handle.setframerate(payload.sample_rate)
        handle.writeframes(payload.frames)
    return buffer.getvalue()


def convert_wav_bytes(
    wav_bytes: bytes,
    *,
    target_sample_rate: int,
    target_channels: int = 1,
    target_sample_width: int = 2,
) -> bytes:
    payload = decode_wav_bytes(wav_bytes)
    frames = payload.frames
    sample_width = payload.sample_width
    channels = payload.channels
    sample_rate = payload.sample_rate

    if sample_width != target_sample_width:
        frames = audioop.lin2lin(frames, sample_width, target_sample_width)
        sample_width = target_sample_width

    if channels == 2 and target_channels == 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    elif channels == 1 and target_channels == 2:
        frames = audioop.tostereo(frames, sample_width, 1.0, 1.0)
        channels = 2
    elif channels != target_channels:
        raise ValueError(
            f"unsupported channel conversion {channels} -> {target_channels}"
        )

    if sample_rate != target_sample_rate:
        frames, _ = audioop.ratecv(
            frames,
            sample_width,
            channels,
            sample_rate,
            target_sample_rate,
            None,
        )
        sample_rate = target_sample_rate

    return encode_wav_bytes(
        WavePayload(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            frames=frames,
        )
    )


def normalized_text_energy(frame_bytes: bytes) -> float:
    if not frame_bytes:
        return 0.0
    rms = audioop.rms(frame_bytes, 2)
    return min(1.0, float(rms) / 32768.0)


class EnergyVad:
    """Simple energy-based VAD with hysteresis and preroll."""

    def __init__(
        self,
        *,
        threshold: float,
        frame_ms: int,
        start_ms: int,
        stop_ms: int,
        min_utterance_ms: int,
        preroll_ms: int,
    ) -> None:
        self.threshold = threshold
        self.frame_ms = frame_ms
        self.start_frames = max(1, start_ms // frame_ms)
        self.stop_frames = max(1, stop_ms // frame_ms)
        self.min_frames = max(1, min_utterance_ms // frame_ms)
        self.preroll_frames = max(0, preroll_ms // frame_ms)
        self._recent: deque[bytes] = deque(maxlen=self.preroll_frames)
        self._active = False
        self._start_hits = 0
        self._stop_hits = 0
        self._frames: list[bytes] = []

    def reset(self) -> None:
        self._recent.clear()
        self._active = False
        self._start_hits = 0
        self._stop_hits = 0
        self._frames = []

    @property
    def active(self) -> bool:
        return self._active

    def process(self, frame_bytes: bytes) -> Optional[bytes]:
        if not frame_bytes:
            return None
        energy = normalized_text_energy(frame_bytes)
        self._recent.append(frame_bytes)
        if not self._active:
            if energy >= self.threshold:
                self._start_hits += 1
            else:
                self._start_hits = 0
            if self._start_hits >= self.start_frames:
                self._active = True
                self._frames = list(self._recent)
                self._stop_hits = 0
            return None

        self._frames.append(frame_bytes)
        if energy < self.threshold:
            self._stop_hits += 1
        else:
            self._stop_hits = 0
        if self._stop_hits >= self.stop_frames:
            utterance = b"".join(self._frames)
            frame_count = len(self._frames)
            self.reset()
            if frame_count >= self.min_frames:
                return utterance
        return None


def encode_pcm16_mono_wav(pcm_frames: bytes, *, sample_rate: int = 16000) -> bytes:
    return encode_wav_bytes(
        WavePayload(
            sample_rate=sample_rate,
            channels=1,
            sample_width=2,
            frames=pcm_frames,
        )
    )


class PlaybackHandle:
    def is_active(self) -> bool:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def wait(self, timeout: Optional[float] = None) -> bool:
        raise NotImplementedError


class SimulatedPlaybackHandle(PlaybackHandle):
    def __init__(self, duration_ms: float) -> None:
        self._duration_s = max(0.0, duration_ms / 1000.0)
        self._done = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        deadline = time.monotonic() + self._duration_s
        while time.monotonic() < deadline:
            if self._stopped.is_set():
                break
            time.sleep(0.01)
        self._done.set()

    def is_active(self) -> bool:
        return not self._done.is_set()

    def stop(self) -> None:
        self._stopped.set()
        self._done.set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout)


class AplayPlaybackHandle(PlaybackHandle):
    def __init__(self, wav_bytes: bytes, *, device: str = "") -> None:
        self._tmp = tempfile.NamedTemporaryFile(prefix="omlx-duplex-", suffix=".wav", delete=False)
        self._tmp.write(wav_bytes)
        self._tmp.flush()
        self._tmp.close()
        command = ["aplay", "-q"]
        if device:
            command.extend(["-D", device])
        command.append(self._tmp.name)
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def is_active(self) -> bool:
        return self._process.poll() is None

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()

    def wait(self, timeout: Optional[float] = None) -> bool:
        try:
            self._process.wait(timeout=timeout)
            return True
        except subprocess.TimeoutExpired:
            return False
        finally:
            try:
                Path(self._tmp.name).unlink(missing_ok=True)
            except OSError:
                pass


class SimulatedPlaybackSink:
    def play(self, wav_bytes: bytes) -> PlaybackHandle:
        payload = decode_wav_bytes(wav_bytes)
        return SimulatedPlaybackHandle(payload.duration_ms)


class AplayPlaybackSink:
    def __init__(self, *, device: str = "") -> None:
        self.device = device

    def play(self, wav_bytes: bytes) -> PlaybackHandle:
        return AplayPlaybackHandle(wav_bytes, device=self.device)


class ARecordCaptureStream:
    """Continuously reads 16kHz mono PCM frames from arecord."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        channels: int = 1,
        frame_ms: int = 20,
        device: str = "",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.device = device
        self.frame_bytes = int(sample_rate * channels * 2 * frame_ms / 1000)
        self._process: Optional[subprocess.Popen[bytes]] = None
        self._queue: "queue.Queue[bytes]" = queue.Queue(maxsize=256)
        self._closed = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._process is not None:
            return
        command = [
            "arecord",
            "-q",
            "-t",
            "raw",
            "-f",
            "S16_LE",
            "-c",
            str(self.channels),
            "-r",
            str(self.sample_rate),
        ]
        if self.device:
            command.extend(["-D", self.device])
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        while not self._closed.is_set():
            chunk = self._process.stdout.read(self.frame_bytes)
            if not chunk:
                break
            try:
                self._queue.put(chunk, timeout=0.1)
            except queue.Full:
                continue
        self._closed.set()

    def read_frame(self, timeout: float = 0.1) -> Optional[bytes]:
        if self._closed.is_set() and self._queue.empty():
            return None
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return b""

    def close(self) -> None:
        self._closed.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
