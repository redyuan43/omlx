# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o-style unattended live self-test harness for duplex experiments."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .asr_client import AsrClient
from .config import DuplexConfig
from .llm_client import LLMChatClient
from .session import DuplexSession, default_playback_sink
from .tts_client import TtsClient


@dataclass(frozen=True)
class ScenarioStep:
    name: str
    prompt_text: str
    post_wait_s: float
    trigger: str = "immediate"
    trigger_playback_active_ms: float | None = None


@dataclass(frozen=True)
class PromptScenario:
    name: str
    steps: List[ScenarioStep]


@dataclass
class ScenarioResult:
    scenario: str
    passed: bool
    turns: int
    barge_in_count: int
    prompt_similarity: float
    assistant_similarity: float
    tail_similarity: float
    speak_latency_ms: float
    session_dir: str
    steps: List[Dict[str, Any]]


PROMPTS: List[PromptScenario] = [
    PromptScenario(
        name="audio_short",
        steps=[
            ScenarioStep(
                name="greeting",
                prompt_text="你好，请用一句简短的话回答我。",
                post_wait_s=2.0,
            )
        ],
    ),
    PromptScenario(
        name="audio_multi_turn_chat",
        steps=[
            ScenarioStep(
                name="turn_1_greeting",
                prompt_text="你好，请先简单介绍一下你自己。",
                post_wait_s=2.0,
            ),
            ScenarioStep(
                name="turn_2_followup",
                prompt_text="你刚才说过你可以陪我聊天。请用一句话重复这个意思。",
                post_wait_s=2.0,
            ),
            ScenarioStep(
                name="turn_3_memory",
                prompt_text="我们刚才已经聊过了。请记住前面的话，再问候我一次。",
                post_wait_s=2.0,
            ),
        ],
    ),
    PromptScenario(
        name="audio_story_interrupt",
        steps=[
            ScenarioStep(
                name="story_start",
                prompt_text="请详细讲一个故事，至少讲五句话，不要只说一句。",
                post_wait_s=1.0,
            ),
            ScenarioStep(
                name="interrupt_question",
                prompt_text="停，先回答我。",
                post_wait_s=2.0,
                trigger="after_assistant_start",
                trigger_playback_active_ms=1200.0,
            ),
            ScenarioStep(
                name="follow_up",
                prompt_text="现在请继续，再补充一句话。",
                post_wait_s=2.0,
            ),
        ],
    ),
]


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", "", text or "")
    normalized = re.sub(r"[，。！？、,.!?；;：“”\"'（）()【】\\[\\]<>《》：:]", "", normalized)
    return normalized.lower()


def _similarity(left: str, right: str) -> float:
    norm_left = _normalize_text(left)
    norm_right = _normalize_text(right)
    if not norm_left or not norm_right:
        return 0.0
    return SequenceMatcher(None, norm_left, norm_right).ratio()


def _tail_similarity(left: str, right: str, *, tail_chars: int = 18) -> float:
    norm_left = _normalize_text(left)
    norm_right = _normalize_text(right)
    return _similarity(norm_left[-tail_chars:], norm_right[-tail_chars:])


class ManagedOmlxServer:
    def __init__(
        self,
        *,
        base_url: str,
        launcher_binary: str,
        model_path: str,
        runtime_root: Path,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.launcher_binary = launcher_binary
        self.model_path = model_path
        self.runtime_root = runtime_root
        self.process: Optional[subprocess.Popen[str]] = None

    @staticmethod
    def detect_default_model_path() -> str:
        env_override = os.getenv("OMLX_DUPLEX_GGUF")
        if env_override:
            return env_override
        candidate = Path(
            "/home/dgx/.lmstudio/models/lmstudio-community/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
        )
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError("could not find default Qwen3.5-4B Q4_K_M GGUF")

    @staticmethod
    def detect_llama_server() -> str:
        env_override = os.getenv("OMLX_DUPLEX_LLAMA_SERVER")
        if env_override:
            return env_override
        which_path = shutil.which("llama-server")
        if which_path:
            return which_path
        candidate = Path("/home/dgx/github/omlx/.runtime/llama.cpp-build/bin/llama-server")
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError("could not find llama-server binary")

    def start(self) -> None:
        if self.process is not None:
            return
        parsed_port = self.base_url.rsplit(":", 1)[-1]
        backend_url = f"http://127.0.0.1:{int(parsed_port) + 20000}"
        base_path = self.runtime_root / "control-plane"
        base_path.mkdir(parents=True, exist_ok=True)
        stub_root = Path("/home/dgx/github/omlx/.runtime/phase4-live/stubs")
        env = os.environ.copy()
        if stub_root.exists():
            pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{stub_root}:/home/dgx/github/omlx:{pythonpath}".rstrip(":")
            )
        stdout_path = base_path / "control-plane.log"
        stderr_path = base_path / "control-plane.err.log"
        stdout_handle = stdout_path.open("w", encoding="utf-8")
        stderr_handle = stderr_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "omlx_dgx.cli",
            "serve",
            "--base-path",
            str(base_path),
            "--backend-kind",
            "llama_cpp",
            "--host",
            "127.0.0.1",
            "--port",
            parsed_port,
            "--backend-url",
            backend_url,
            "--launcher-binary",
            self.launcher_binary,
            "--artifact-path",
            self.model_path,
            "--model-source",
            "gguf",
            "--quant-mode",
            "gguf_experimental",
            "--gguf-variant",
            "Q4_K_M",
            "--serving-preset",
            "single_session_low_latency",
            "--model-id",
            "qwen35-4b",
            "--model-alias",
            "qwen35-4b",
        ]
        self.process = subprocess.Popen(
            command,
            cwd=str(base_path),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=env,
        )
        self._wait_for_control_plane()
        requests.post(
            f"{self.base_url}/admin/api/runtime/start",
            timeout=120,
        ).raise_for_status()
        self._wait_for_models()

    def stop(self) -> None:
        try:
            requests.post(f"{self.base_url}/admin/api/runtime/stop", timeout=30)
        except Exception:
            pass
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None

    def _wait_for_control_plane(self) -> None:
        deadline = time.monotonic() + 120.0
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.ok:
                    return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("timed out waiting for temporary omlx control-plane")

    def _wait_for_models(self) -> None:
        deadline = time.monotonic() + 180.0
        while time.monotonic() < deadline:
            try:
                response = requests.get(f"{self.base_url}/v1/models", timeout=5)
                if response.ok:
                    payload = response.json()
                    if payload.get("data"):
                        return
            except Exception:
                pass
            time.sleep(1.0)
        raise RuntimeError("timed out waiting for temporary omlx models endpoint")


class DuplexSelfTester:
    def __init__(
        self,
        config: DuplexConfig,
        *,
        bootstrap_omlx: bool = True,
        real_playback: bool = False,
        play_user_prompts: bool = False,
    ) -> None:
        self.config = config
        self.bootstrap_omlx = bootstrap_omlx
        self.real_playback = real_playback
        self.play_user_prompts = play_user_prompts
        self.runtime_root = config.runtime_path()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self.managed_server: Optional[ManagedOmlxServer] = None
        self.asr_client = AsrClient(config.services.asr_url, timeout=config.services.request_timeout_s)
        self.tts_client = TtsClient(config.services.tts_url, timeout=config.services.request_timeout_s)
        self.llm_client = LLMChatClient(config.services.llm_base_url, timeout=config.services.request_timeout_s)

    def ensure_services(self) -> None:
        self.asr_client.health()
        self.tts_client.ensure_loaded(timeout_s=300.0)
        try:
            self.llm_client.health()
        except Exception:
            if not self.bootstrap_omlx:
                raise
            self.managed_server = ManagedOmlxServer(
                base_url=self.config.services.llm_base_url,
                launcher_binary=ManagedOmlxServer.detect_llama_server(),
                model_path=ManagedOmlxServer.detect_default_model_path(),
                runtime_root=self.runtime_root,
            )
            self.managed_server.start()
            self.llm_client.health()

    def close(self) -> None:
        if self.managed_server is not None:
            self.managed_server.stop()
            self.managed_server = None

    def synth_prompt_audio(self, text: str, *, trace_id: str) -> bytes:
        tts = self.tts_client.speak(
            text,
            speaker=self.config.services.speaker,
            speed=1.0,
            trace_id=trace_id,
        )
        return tts.wav_bytes

    def transcribe_assistant_audio(self, wav_bytes: bytes, *, trace_id: str) -> str:
        result = self.asr_client.transcribe_wav_bytes(wav_bytes, trace_id=trace_id)
        return result.text

    def run_scenario(self, scenario: PromptScenario) -> ScenarioResult:
        session_config = DuplexConfig(
            runtime_dir=str(self.runtime_root / "sessions"),
            audio=self.config.audio,
            services=self.config.services,
            selftest=self.config.selftest,
        )
        session = DuplexSession(
            config=session_config,
            asr_client=self.asr_client,
            tts_client=self.tts_client,
            llm_client=self.llm_client,
            playback_sink=default_playback_sink(
                simulated=not self.real_playback,
                device=self.config.audio.playback_device,
            ),
        )
        try:
            step_results: List[Dict[str, Any]] = []
            previous_context = None
            for index, step in enumerate(scenario.steps):
                prompt_audio = self.synth_prompt_audio(
                    step.prompt_text,
                    trace_id=f"{session.session_id}-{step.name}-prompt",
                )
                prompt_playback = None
                if self.real_playback and self.play_user_prompts:
                    prompt_playback = default_playback_sink(
                        simulated=False,
                        device=self.config.audio.playback_device,
                    ).play(prompt_audio)
                if step.trigger == "after_assistant_start":
                    if previous_context is None or previous_context.playback is None:
                        raise RuntimeError("interrupt step requires an active previous playback")
                    deadline = time.monotonic() + 15.0
                    while previous_context.playback.is_active() and time.monotonic() < deadline:
                        break
                    if step.trigger_playback_active_ms:
                        time.sleep(step.trigger_playback_active_ms / 1000.0)
                    session.interrupt_playback(reason="selftest_barge_in")
                context = session.submit_user_audio(prompt_audio, prompt_label=step.name)
                if prompt_playback is not None and step.trigger != "after_assistant_start":
                    prompt_playback.wait(timeout=15.0)
                context.wait_complete(timeout=max(30.0, self.config.services.request_timeout_s))
                result = context.result
                assistant_audio = Path(result.reply_audio_path).read_bytes()
                assistant_transcript = self.transcribe_assistant_audio(
                    assistant_audio,
                    trace_id=f"{session.session_id}-{step.name}-assistant-asr",
                )
                result.assistant_transcript = assistant_transcript
                Path(result.user_audio_path).with_name("turn.json").write_text(
                    json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                step_results.append(
                    {
                        "turn_index": result.turn_index,
                        "name": step.name,
                        "prompt_text": step.prompt_text,
                        "prompt_transcript": result.transcript,
                        "assistant_text": result.reply_text,
                        "assistant_transcript": assistant_transcript,
                        "prompt_similarity": _similarity(step.prompt_text, result.transcript),
                        "assistant_similarity": _similarity(result.reply_text, assistant_transcript),
                        "tail_similarity": _tail_similarity(result.reply_text, assistant_transcript),
                        "speak_latency_ms": result.speak_latency_ms,
                        "interrupted": result.interrupted,
                    }
                )
                previous_context = context
                next_step = scenario.steps[index + 1] if index + 1 < len(scenario.steps) else None
                if next_step is not None and next_step.trigger == "after_assistant_start":
                    continue
                context.playback.wait(timeout=max(5.0, step.post_wait_s + 5.0))

            for item in step_results:
                turn_index = int(item["turn_index"])
                item["interrupted"] = session.turns[turn_index - 1].interrupted
            prompt_similarity = min(item["prompt_similarity"] for item in step_results)
            completed_steps = [item for item in step_results if not item["interrupted"]]
            compare_steps = completed_steps or step_results
            assistant_similarity = min(item["assistant_similarity"] for item in compare_steps)
            tail_similarity = min(item["tail_similarity"] for item in compare_steps)
            passed = (
                prompt_similarity >= self.config.selftest.prompt_similarity
                and assistant_similarity >= self.config.selftest.assistant_similarity
                and tail_similarity >= self.config.selftest.tail_similarity
            )
            if scenario.name == "audio_story_interrupt":
                passed = passed and session.interrupt_count >= self.config.selftest.min_barge_in_count
            scenario_result = ScenarioResult(
                scenario=scenario.name,
                passed=passed,
                turns=len(step_results),
                barge_in_count=session.interrupt_count,
                prompt_similarity=prompt_similarity,
                assistant_similarity=assistant_similarity,
                tail_similarity=tail_similarity,
                speak_latency_ms=min(item["speak_latency_ms"] for item in step_results),
                session_dir=str(session.runtime_dir),
                steps=step_results,
            )
            Path(session.runtime_dir / "scenario.json").write_text(
                json.dumps(asdict(scenario_result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return scenario_result
        finally:
            session.close()

    def run(self, *, scenarios: Optional[List[str]] = None) -> Dict[str, Any]:
        selected = [item for item in PROMPTS if not scenarios or item.name in scenarios]
        self.ensure_services()
        results = []
        try:
            for scenario in selected:
                results.append(asdict(self.run_scenario(scenario)))
        finally:
            self.close()
        payload = {
            "name": "omlx-duplex-mini-cpmo-style-selftest",
            "passed": all(item["passed"] for item in results),
            "scenarios": results,
        }
        report_path = self.runtime_root / f"selftest-{int(time.time())}.json"
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        payload["report_path"] = str(report_path)
        return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run unattended duplex live self tests")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--asr-url", default="http://127.0.0.1:8001")
    parser.add_argument("--tts-url", default="http://127.0.0.1:8002")
    parser.add_argument("--model", default="qwen35-4b")
    parser.add_argument("--runtime-dir", default=".runtime/duplex-selftest")
    parser.add_argument("--speaker", default="")
    parser.add_argument("--playback-device", default="")
    parser.add_argument("--real-playback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--play-user-prompts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bootstrap-omlx", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scenario", action="append", default=[])
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = DuplexConfig(runtime_dir=args.runtime_dir)
    config.services.llm_base_url = args.llm_base_url
    config.services.asr_url = args.asr_url
    config.services.tts_url = args.tts_url
    config.services.model = args.model
    config.services.speaker = args.speaker
    config.audio.playback_device = args.playback_device
    tester = DuplexSelfTester(
        config,
        bootstrap_omlx=bool(args.bootstrap_omlx),
        real_playback=bool(args.real_playback),
        play_user_prompts=bool(args.play_user_prompts),
    )
    payload = tester.run(scenarios=args.scenario or None)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
