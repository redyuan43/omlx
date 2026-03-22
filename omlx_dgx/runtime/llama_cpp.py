# SPDX-License-Identifier: Apache-2.0
"""llama.cpp server adapter with managed process lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import urlparse

from omlx.api.thinking import extract_thinking
from omlx_dgx.config import (
    BackendConfig,
    DGXRuntimeConfig,
    LlamaCppModelRegistration,
    apply_llama_cpp_serving_preset,
)
from omlx_dgx.model_capabilities import infer_model_capabilities, infer_multimodal_capabilities
from omlx_dgx.session_restore import (
    SessionRestoreSnapshot,
    SessionRestoreStore,
    snapshot_key,
)

from .backend import (
    BackendAdapter,
    BackendCapabilities,
    BackendError,
    ExternalOpenAIModelAdapter,
    HttpOpenAIBackendAdapter,
    RuntimeMetrics,
)


def _stringify_command(args: List[str]) -> str:
    return shlex.join(args)


_RECYCLE_OWNED_IDLE_SLOT_MIN_AGE_SECONDS = 1.0
_SINGLE_SLOT_CONTINUATION_TTL_SECONDS = 600.0
_PERSISTED_RUNTIME_STATE_VERSION = 1
_SESSION_RESTORE_RECENT_LIMIT = 16
_CACHE_REUSE_UNSUPPORTED_TOKENS = (
    "qwen3.5",
    "qwen35",
    "qwen3_5",
    "qwen3-5",
)


def _coerce_response_payload(response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    if "application/json" in content_type:
        return response.json()
    return {
        "ok": response.ok,
        "status_code": response.status_code,
        "text": response.text.strip(),
    }


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type == "text":
                    parts.append(str(item.get("text", "")))
                elif item_type == "image_url":
                    parts.append("[image]")
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _extract_prompt_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("messages"), list):
        parts: List[str] = []
        for message in payload["messages"]:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = _coerce_text(message.get("content"))
            if content:
                parts.append(f"{role}:{content}")
        return "\n".join(parts)
    prompt = payload.get("prompt")
    if isinstance(prompt, list):
        return "\n".join(str(item) for item in prompt)
    return _coerce_text(prompt)


def _estimate_prompt_tokens(payload: Dict[str, Any]) -> int:
    text = _extract_prompt_text(payload)
    if not text:
        return 0
    chars_estimate = max(1, len(text) // 4)
    words_estimate = len(text.split())
    return max(chars_estimate, words_estimate)


def _extract_routing_key(payload: Dict[str, Any]) -> Optional[str]:
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        for key_name in ("omlx_conversation_key", "conversation_id", "session_id"):
            value = metadata.get(key_name)
            if value:
                return str(value)
    if payload.get("user"):
        return str(payload["user"])
    return None


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _canonical_messages(payload: Dict[str, Any]) -> tuple[str, ...]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    normalized: List[str] = []
    for message in messages:
        if isinstance(message, dict):
            normalized.append(_stable_json(message))
        else:
            normalized.append(str(message))
    return tuple(normalized)


def _canonical_prompt(payload: Dict[str, Any]) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if prompt is None:
        return ""
    if isinstance(prompt, list):
        return _stable_json(prompt)
    if isinstance(prompt, dict):
        return _stable_json(prompt)
    return str(prompt)


def _canonical_message(message: Any) -> str:
    if isinstance(message, dict):
        return _stable_json(message)
    return str(message)


def _canonical_message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "") or "")
    if isinstance(message, str):
        try:
            payload = json.loads(message)
        except Exception:
            return ""
        if isinstance(payload, dict):
            return str(payload.get("role", "") or "")
    return ""


def _response_assistant_message(response_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    message = choice.get("message")
    if not isinstance(message, dict):
        return None
    normalized = dict(message)
    normalized.setdefault("role", "assistant")
    return normalized


def _continuation_shape_digest(path: str, payload: Dict[str, Any]) -> str:
    shape = {
        "path": path,
        "model": payload.get("model"),
        "chat_template_kwargs": payload.get("chat_template_kwargs"),
        "tools": payload.get("tools"),
        "tool_choice": payload.get("tool_choice"),
        "enableThinking": payload.get("enableThinking"),
        "reasoning": payload.get("reasoning"),
        "reasoning_budget": payload.get("reasoning_budget"),
        "reasoning_format": payload.get("reasoning_format"),
        "thinking_forced_open": payload.get("thinking_forced_open"),
        "response_format": payload.get("response_format"),
    }
    return hashlib.sha256(_stable_json(shape).encode("utf-8")).hexdigest()[:16]


def _disable_thinking_requested(payload: Dict[str, Any]) -> bool:
    if payload.get("think") is False:
        return True
    chat_template_kwargs = payload.get("chat_template_kwargs")
    if isinstance(chat_template_kwargs, dict) and chat_template_kwargs.get("enable_thinking") is False:
        return True
    if payload.get("enableThinking") is False:
        return True
    if payload.get("reasoning") is False:
        return True
    if payload.get("reasoning_budget") == 0:
        return True
    if payload.get("thinking_forced_open") is False:
        return True
    return False


def _infer_gguf_variant(*model_refs: str) -> str:
    pattern = re.compile(r"(IQ\d+_[A-Z0-9_]+|Q\d+_[A-Z0-9_]+)", re.IGNORECASE)
    for model_ref in model_refs:
        if not model_ref:
            continue
        match = pattern.search(Path(model_ref).name)
        if match:
            return match.group(1).upper()
    return ""


def _artifact_summary(model_ref: str, *, model_source: str, gguf_variant: str) -> Dict[str, Any]:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        if candidate.is_file():
            artifact_kind = "local_file"
        elif candidate.is_dir():
            artifact_kind = "local_dir"
        else:
            artifact_kind = "local_other"
        artifact_exists: Optional[bool] = True
    else:
        artifact_kind = "remote_ref"
        artifact_exists = None
    return {
        "model_source": model_source,
        "artifact_exists": artifact_exists,
        "artifact_kind": artifact_kind,
        "gguf_variant": gguf_variant,
        "quantization_hint": gguf_variant or "unknown",
    }


def _binary_exists(binary: str) -> bool:
    if "/" in binary:
        return Path(binary).expanduser().exists()
    return shutil.which(binary) is not None


def _cache_reuse_model_blocker(*model_refs: str) -> str:
    normalized = " ".join(str(model_ref or "").lower() for model_ref in model_refs if model_ref)
    if any(token in normalized for token in _CACHE_REUSE_UNSUPPORTED_TOKENS):
        return (
            "Qwen3.5 uses the hybrid GDN/Mamba-recurrent cache path; "
            "llama.cpp cache_reuse is not supported"
        )
    return ""


def _cache_reuse_runtime_blocker(log_lines: List[str]) -> str:
    for line in log_lines:
        lowered = line.lower()
        if "cache_reuse is not supported" in lowered or "cache reuse is not supported" in lowered:
            return "llama.cpp reported this runtime context does not support cache_reuse"
    return ""


def _strip_thinking_content(text: str) -> str:
    thinking, regular = extract_thinking(text)
    if thinking:
        return regular.strip()
    if "<think>" in text and "</think>" in text:
        return regular.strip()
    return text


def _merge_extra_body(payload: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = payload.get("extra_body")
    if not isinstance(extra_body, dict):
        return payload
    merged: Dict[str, Any] = dict(extra_body)
    for key, value in payload.items():
        if key == "extra_body":
            continue
        if (
            key == "chat_template_kwargs"
            and isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            combined = dict(merged[key])
            combined.update(value)
            merged[key] = combined
            continue
        merged[key] = value
    return merged


@dataclass
class LlamaCppDiagnostics:
    adapter: str
    backend_format: str
    base_url: str
    launcher_binary: str
    launcher_binary_exists: bool
    model_repo_id: str
    quant_mode: str
    model_source: str
    artifact_path: str
    effective_model_path: str
    model_flag: str
    serving_preset: str
    artifact_summary: Dict[str, Any]
    gguf_variant: str
    ctx_size: int
    parallel_slots: int
    n_gpu_layers: int
    flash_attn: bool
    batch_size: int
    ubatch_size: int
    cache_ram_mib: int
    configured_cache_reuse: int
    cache_reuse: int
    cache_reuse_supported: bool
    cache_reuse_blocker: str
    checkpoint_every_n_tokens: int
    ctx_checkpoints: int
    slot_prompt_similarity: float
    enable_runtime_metrics: bool
    enable_session_stickiness: bool
    enable_session_restore: bool
    session_restore_min_prompt_tokens: int
    sticky_session_prompt_threshold: int
    sticky_max_sessions: int
    split_mode: str
    no_context_shift: bool
    jinja: bool
    reasoning_format: str
    launcher_cmd: str
    launcher_cmd_error: str
    managed_pid: Optional[int]
    managed_process_running: bool
    startup_timeout_seconds: int
    single_session_continuation_enabled: bool
    single_session_continuation_ttl_seconds: int
    slot_save_path: str
    supports_embeddings: bool
    supports_rerank: bool
    supports_vision: bool
    supports_ocr: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SlotRouteDecision:
    reason: str
    path: str
    slot_id: Optional[int]
    estimated_prompt_tokens: int
    sticky_key_hash: str
    cache_prompt: bool
    explicit_slot: bool
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = round(self.timestamp, 3)
        return payload


@dataclass
class SessionBinding:
    slot_id: int
    estimated_prompt_tokens: int
    last_used_at: float
    sticky_key_hash: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "last_used_at": round(self.last_used_at, 3),
            "sticky_key_hash": self.sticky_key_hash,
        }


@dataclass
class ContinuationState:
    conversation_id: str
    slot_id: int
    model_id: str
    estimated_prompt_tokens: int
    last_used_at: float
    prefix_digest: str
    request_shape_digest: str
    prompt_mode: str
    prompt_value: str | tuple[str, ...]
    message_count: int
    slot_prompt_value: str | tuple[str, ...]
    slot_message_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_key_hash": _hash_key(self.conversation_id),
            "slot_id": self.slot_id,
            "model_id": self.model_id,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "last_used_at": round(self.last_used_at, 3),
            "prefix_digest": self.prefix_digest,
            "request_shape_digest": self.request_shape_digest,
            "prompt_mode": self.prompt_mode,
            "message_count": self.message_count,
            "slot_message_count": self.slot_message_count,
        }


@dataclass
class ContinuationDecision:
    reason: str
    conversation_key_hash: str
    slot_id: int
    estimated_prompt_tokens: int
    continuation_hit: bool
    prefix_drift: bool
    suffix_only: bool
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = round(self.timestamp, 3)
        return payload


class LlamaCppProcessManager:
    """Manages a llama-server child process."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.pid_path = self.state_dir / "llama.pid"
        self.log_path = self.state_dir / "llama.log"
        self.cmd_path = self.state_dir / "llama.cmd"

    def _read_pid(self) -> Optional[int]:
        if not self.pid_path.exists():
            return None
        try:
            return int(self.pid_path.read_text(encoding="utf-8").strip())
        except Exception:
            return None

    def is_running(self) -> bool:
        pid = self._read_pid()
        if pid is None:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def pid(self) -> Optional[int]:
        if self.is_running():
            return self._read_pid()
        return None

    def start(self, args: List[str], *, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if self.is_running():
            return {"started": False, "pid": self.pid(), "reason": "already_running"}

        with self.log_path.open("ab") as log_file:
            process = subprocess.Popen(  # noqa: S603
                args,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.pid_path.write_text(str(process.pid), encoding="utf-8")
        self.cmd_path.write_text(_stringify_command(args), encoding="utf-8")
        return {
            "started": True,
            "pid": process.pid,
            "command": _stringify_command(args),
        }

    def stop(self) -> Dict[str, Any]:
        pid = self.pid()
        if pid is None:
            return {"stopped": False, "reason": "not_running"}
        os.killpg(pid, signal.SIGTERM)
        if self.pid_path.exists():
            self.pid_path.unlink()
        return {"stopped": True, "pid": pid}

    def logs(self, lines: int = 40) -> Dict[str, Any]:
        content: List[str] = []
        if self.log_path.exists():
            content = self.log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        return {
            "lines": content[-lines:],
            "path": str(self.log_path),
            "command": self.command(),
        }

    def command(self) -> str:
        if not self.cmd_path.exists():
            return ""
        return self.cmd_path.read_text(encoding="utf-8").strip()


class LlamaCppBackendAdapter(HttpOpenAIBackendAdapter):
    """Runtime-aware adapter for llama.cpp's OpenAI-compatible server."""

    def __init__(self, config: BackendConfig, state_dir: Path) -> None:
        super().__init__(config.base_url, timeout=120.0)
        self.config = config
        self.state_dir = Path(state_dir)
        self.process_manager = LlamaCppProcessManager(self.state_dir / "runtime")
        self._slot_save_path = self.state_dir / "runtime" / "slot_saves"
        self._slot_save_path.mkdir(parents=True, exist_ok=True)
        self._session_restore_store = SessionRestoreStore(
            self.state_dir / "runtime" / "session_restore"
        )
        parsed = urlparse(config.base_url)
        self.launch_host = parsed.hostname or "127.0.0.1"
        self.launch_port = parsed.port or 30000
        self._lock = threading.RLock()
        self._session_bindings: "OrderedDict[str, SessionBinding]" = OrderedDict()
        self._slot_owners: Dict[int, str] = {}
        self._recent_slot_decisions: Deque[SlotRouteDecision] = deque(maxlen=32)
        self._last_slot_decision: Optional[SlotRouteDecision] = None
        self._single_slot_continuation: Optional[ContinuationState] = None
        self._single_slot_continuation_dirty = False
        self._single_slot_recovery_pending = False
        self._single_slot_restored_from_disk = False
        self._recent_continuation_decisions: Deque[ContinuationDecision] = deque(maxlen=32)
        self._last_continuation_decision: Optional[ContinuationDecision] = None
        self._runtime_state_signature = ""
        self._runtime_state_path = self.state_dir / "llama_cpp_runtime_state.json"
        self._last_slot_save: Optional[Dict[str, Any]] = None
        self._last_slot_restore: Optional[Dict[str, Any]] = None
        self._session_restore_counts: Dict[str, int] = {
            "saved": 0,
            "restored": 0,
            "restore_miss": 0,
            "save_error": 0,
            "restore_error": 0,
            "restore_fallback": 0,
        }
        self._continuation_counts: Dict[str, int] = {
            "disabled": 0,
            "no_conversation_id": 0,
            "cold_start": 0,
            "hit": 0,
            "recovery_replay": 0,
            "prefix_drift": 0,
            "shape_changed": 0,
            "model_changed": 0,
            "expired": 0,
            "slot_reset": 0,
        }
        self._slot_route_counts: Dict[str, int] = {
            "pass_through": 0,
            "explicit_slot": 0,
            "sticky_existing": 0,
            "sticky_new_long_prompt": 0,
            "sticky_restored": 0,
            "sticky_no_idle_slot": 0,
            "short_prompt": 0,
            "short_prompt_unowned_slot": 0,
            "short_prompt_recycled_slot": 0,
            "no_routing_key": 0,
            "unkeyed_idle_slot": 0,
            "unkeyed_recycled_slot": 0,
            "single_slot_cold_start": 0,
            "single_slot_hit": 0,
            "single_slot_recovery_replay": 0,
            "single_slot_prefix_drift": 0,
            "single_slot_shape_changed": 0,
            "single_slot_model_changed": 0,
            "single_slot_expired": 0,
            "single_slot_slot_reset": 0,
            "single_slot_no_conversation_id": 0,
        }
        self._load_runtime_state()

    @classmethod
    def from_backend_config(
        cls, config: BackendConfig, state_dir: str | Path
    ) -> "LlamaCppBackendAdapter":
        return cls(config, Path(state_dir))

    def _effective_model_path(self) -> str:
        return self.config.artifact_path or self.config.model_repo_id

    def _backend_format(self) -> str:
        return "llama_cpp_gguf"

    def _gguf_variant(self) -> str:
        return self.config.gguf_variant or _infer_gguf_variant(
            self.config.artifact_path,
            self.config.model_repo_id,
        )

    def _cache_reuse_model_blocker(self) -> str:
        return _cache_reuse_model_blocker(
            self._effective_model_path(),
        )

    def _cache_reuse_runtime_blocker(self) -> str:
        if not self.process_manager.log_path.exists():
            return ""
        try:
            log_lines = self.process_manager.logs(lines=160)["lines"]
        except Exception:
            return ""
        return _cache_reuse_runtime_blocker(log_lines)

    def _effective_cache_reuse(self) -> int:
        configured = max(0, int(self.config.cache_reuse or 0))
        if configured <= 0:
            return 0
        if self._cache_reuse_model_blocker():
            return 0
        if self._cache_reuse_runtime_blocker():
            return 0
        return configured

    def _cache_reuse_blocker(self) -> str:
        model_blocker = self._cache_reuse_model_blocker()
        if model_blocker:
            return model_blocker
        return self._cache_reuse_runtime_blocker()

    def _model_flag(self, model_path: str) -> str:
        candidate = Path(model_path).expanduser()
        if candidate.exists():
            return "--model"
        if self.config.model_source == "gguf":
            return "-hf"
        return "--model"

    def _build_launch_command(self) -> List[str]:
        if self.config.launcher_cmd:
            return shlex.split(self.config.launcher_cmd)

        model_path = self._effective_model_path()
        if not model_path:
            raise BackendError("artifact_path or model_repo_id is required for llama.cpp")

        args = [
            self.config.launcher_binary,
            self._model_flag(model_path),
            model_path,
            "--host",
            self.launch_host,
            "--port",
            str(self.launch_port),
            "--ctx-size",
            str(self.config.ctx_size),
            "--parallel",
            str(self.config.parallel_slots),
            "--n-gpu-layers",
            str(self.config.n_gpu_layers),
            "--batch-size",
            str(self.config.batch_size),
            "--ubatch-size",
            str(self.config.ubatch_size),
            "--cache-ram",
            str(self.config.cache_ram_mib),
            "--checkpoint-every-n-tokens",
            str(self.config.checkpoint_every_n_tokens),
            "--ctx-checkpoints",
            str(self.config.ctx_checkpoints),
            "--slot-prompt-similarity",
            str(self.config.slot_prompt_similarity),
            "--split-mode",
            self.config.split_mode,
        ]
        if self.config.mmproj_path:
            args.extend(["--mmproj", self.config.mmproj_path])
        args.extend(["--flash-attn", "on" if self.config.flash_attn else "off"])
        effective_cache_reuse = self._effective_cache_reuse()
        if effective_cache_reuse > 0:
            args.extend(["--cache-reuse", str(effective_cache_reuse)])
        if self.config.enable_runtime_metrics:
            args.append("--metrics")
        model_capabilities = infer_model_capabilities(
            self.config.artifact_path,
            self.config.model_repo_id,
        )
        if model_capabilities.rerank:
            args.append("--reranking")
        if self.config.enable_session_restore:
            args.extend(["--slot-save-path", str(self._slot_save_path)])
        if self.config.no_context_shift:
            args.append("--no-context-shift")
        if self.config.jinja:
            args.append("--jinja")
        if self.config.reasoning_format:
            args.extend(["--reasoning-format", self.config.reasoning_format])
        return args

    def diagnostics(self) -> LlamaCppDiagnostics:
        launcher_cmd = ""
        launcher_cmd_error = ""
        effective_model_path = self._effective_model_path()
        try:
            launcher_cmd = _stringify_command(self._build_launch_command())
        except BackendError as exc:
            launcher_cmd_error = str(exc)
        gguf_variant = self._gguf_variant()
        cache_reuse_blocker = self._cache_reuse_blocker()
        effective_cache_reuse = self._effective_cache_reuse()
        model_capabilities = infer_model_capabilities(
            self.config.artifact_path,
            self.config.model_repo_id,
        )
        return LlamaCppDiagnostics(
            adapter="llama_cpp",
            backend_format=self._backend_format(),
            base_url=self.base_url,
            launcher_binary=self.config.launcher_binary,
            launcher_binary_exists=_binary_exists(self.config.launcher_binary),
            model_repo_id=self.config.model_repo_id,
            quant_mode=self.config.quant_mode,
            model_source=self.config.model_source,
            artifact_path=self.config.artifact_path,
            effective_model_path=effective_model_path,
            model_flag=self._model_flag(effective_model_path),
            serving_preset=self.config.serving_preset,
            artifact_summary=_artifact_summary(
                effective_model_path,
                model_source=self.config.model_source,
                gguf_variant=gguf_variant,
            ),
            gguf_variant=gguf_variant,
            ctx_size=self.config.ctx_size,
            parallel_slots=self.config.parallel_slots,
            n_gpu_layers=self.config.n_gpu_layers,
            flash_attn=self.config.flash_attn,
            batch_size=self.config.batch_size,
            ubatch_size=self.config.ubatch_size,
            cache_ram_mib=self.config.cache_ram_mib,
            configured_cache_reuse=self.config.cache_reuse,
            cache_reuse=effective_cache_reuse,
            cache_reuse_supported=not bool(cache_reuse_blocker),
            cache_reuse_blocker=cache_reuse_blocker,
            checkpoint_every_n_tokens=self.config.checkpoint_every_n_tokens,
            ctx_checkpoints=self.config.ctx_checkpoints,
            slot_prompt_similarity=self.config.slot_prompt_similarity,
            enable_runtime_metrics=self.config.enable_runtime_metrics,
            enable_session_stickiness=self.config.enable_session_stickiness,
            enable_session_restore=self.config.enable_session_restore,
            session_restore_min_prompt_tokens=self._session_restore_threshold(),
            sticky_session_prompt_threshold=self.config.sticky_session_prompt_threshold,
            sticky_max_sessions=self.config.sticky_max_sessions,
            split_mode=self.config.split_mode,
            no_context_shift=self.config.no_context_shift,
            jinja=self.config.jinja,
            reasoning_format=self.config.reasoning_format,
            launcher_cmd=launcher_cmd,
            launcher_cmd_error=launcher_cmd_error,
            managed_pid=self.process_manager.pid(),
            managed_process_running=self.process_manager.is_running(),
            startup_timeout_seconds=self.config.startup_timeout_seconds,
            single_session_continuation_enabled=(
                self.config.enable_session_stickiness and self.config.parallel_slots <= 1
            ),
            single_session_continuation_ttl_seconds=int(
                _SINGLE_SLOT_CONTINUATION_TTL_SECONDS
            ),
            slot_save_path=str(self._slot_save_path),
            supports_embeddings=model_capabilities.embeddings,
            supports_rerank=model_capabilities.rerank,
            supports_vision=gguf_variant.startswith("VL")
            or model_capabilities.vision_chat,
            supports_ocr=model_capabilities.ocr,
        )

    def _extract_slots_list(self, payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        data = payload.get("data")
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        slots = payload.get("slots")
        if isinstance(slots, list):
            return [item for item in slots if isinstance(item, dict)]
        return []

    def _slot_summary(self, slots_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for slot in self._extract_slots_list(slots_payload):
            next_token = slot.get("next_token")
            token_state = next_token[0] if isinstance(next_token, list) and next_token else {}
            summary.append(
                {
                    "id": slot.get("id"),
                    "is_processing": slot.get("is_processing"),
                    "n_ctx": slot.get("n_ctx"),
                    "id_task": slot.get("id_task"),
                    "n_decoded": token_state.get("n_decoded"),
                    "n_remain": token_state.get("n_remain"),
                }
            )
        return summary

    def _current_runtime_signature(self) -> str:
        pid = self.process_manager.pid()
        if pid is not None:
            return f"managed-pid:{pid}"
        return ""

    def _serialize_prompt_value(
        self, value: str | tuple[str, ...]
    ) -> str | List[str]:
        if isinstance(value, tuple):
            return list(value)
        return value

    def _deserialize_prompt_value(
        self,
        *,
        prompt_mode: str,
        value: Any,
    ) -> str | tuple[str, ...]:
        if prompt_mode == "messages":
            if isinstance(value, list):
                return tuple(str(item) for item in value)
            if isinstance(value, tuple):
                return tuple(str(item) for item in value)
            if isinstance(value, str) and value:
                return (value,)
            return ()
        if value is None:
            return ""
        return str(value)

    def _serialize_continuation_state(
        self,
        state: ContinuationState,
    ) -> Dict[str, Any]:
        payload = asdict(state)
        payload["prompt_value"] = self._serialize_prompt_value(state.prompt_value)
        payload["slot_prompt_value"] = self._serialize_prompt_value(
            state.slot_prompt_value
        )
        return payload

    def _deserialize_continuation_state(
        self,
        payload: Dict[str, Any],
    ) -> Optional[ContinuationState]:
        try:
            prompt_mode = str(payload.get("prompt_mode", "prompt"))
            return ContinuationState(
                conversation_id=str(payload["conversation_id"]),
                slot_id=int(payload.get("slot_id", 0)),
                model_id=str(payload.get("model_id", "")),
                estimated_prompt_tokens=int(payload.get("estimated_prompt_tokens", 0)),
                last_used_at=float(payload.get("last_used_at", 0.0)),
                prefix_digest=str(payload.get("prefix_digest", "")),
                request_shape_digest=str(payload.get("request_shape_digest", "")),
                prompt_mode=prompt_mode,
                prompt_value=self._deserialize_prompt_value(
                    prompt_mode=prompt_mode,
                    value=payload.get("prompt_value"),
                ),
                message_count=int(payload.get("message_count", 0)),
                slot_prompt_value=self._deserialize_prompt_value(
                    prompt_mode=prompt_mode,
                    value=payload.get("slot_prompt_value"),
                ),
                slot_message_count=int(payload.get("slot_message_count", 0)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def _session_restore_enabled(self) -> bool:
        return bool(
            self.config.enable_session_restore
            and self.config.enable_session_stickiness
        )

    def _session_restore_threshold(self) -> int:
        threshold = (
            self.config.session_restore_min_prompt_tokens
            or self.config.sticky_session_prompt_threshold
            or 1
        )
        return max(1, int(threshold))

    def _session_restore_filename(
        self,
        *,
        conversation_id: str,
        model_id: str,
        slot_id: int,
    ) -> str:
        return f"{snapshot_key(model_id, conversation_id)}-slot{slot_id}.bin"

    def _slot_action(
        self,
        *,
        slot_id: int,
        action: str,
        payload: Dict[str, Any],
        timeout: float = 180.0,
    ) -> Dict[str, Any]:
        response = self._request(
            "POST",
            f"slots/{slot_id}?action={action}",
            timeout=timeout,
            json=payload,
        )
        if not response.ok:
            raise BackendError(
                f"llama.cpp slot {action} failed for slot {slot_id}: "
                f"{response.status_code} {response.text.strip()}"
            )
        result = _coerce_response_payload(response)
        if not isinstance(result, dict):
            raise BackendError(
                f"llama.cpp slot {action} returned a non-json payload"
            )
        return result

    def _should_persist_session_snapshot(
        self,
        *,
        conversation_id: str,
        model_id: str,
        estimated_prompt_tokens: int,
    ) -> bool:
        if not self._session_restore_enabled():
            return False
        if self.config.parallel_slots <= 1:
            return True
        if estimated_prompt_tokens >= self._session_restore_threshold():
            return True
        existing = self._session_restore_store.get(
            conversation_id=conversation_id,
            model_id=model_id,
            touch=False,
        )
        return existing is not None

    def _persist_session_snapshot(
        self,
        state: ContinuationState,
    ) -> bool:
        if not self._should_persist_session_snapshot(
            conversation_id=state.conversation_id,
            model_id=state.model_id,
            estimated_prompt_tokens=state.estimated_prompt_tokens,
        ):
            return False

        existing = self._session_restore_store.get(
            conversation_id=state.conversation_id,
            model_id=state.model_id,
            touch=False,
        )
        filename = (
            existing.save_filename
            if existing is not None and existing.save_filename
            else self._session_restore_filename(
                conversation_id=state.conversation_id,
                model_id=state.model_id,
                slot_id=state.slot_id,
            )
        )
        try:
            result = self._slot_action(
                slot_id=state.slot_id,
                action="save",
                payload={"filename": filename},
            )
        except Exception as exc:
            with self._lock:
                self._session_restore_counts["save_error"] = (
                    self._session_restore_counts.get("save_error", 0) + 1
                )
                self._last_slot_save = {
                    "ok": False,
                    "conversation_key_hash": _hash_key(state.conversation_id),
                    "slot_id": state.slot_id,
                    "filename": filename,
                    "error": str(exc),
                    "timestamp": round(time.time(), 3),
                }
            return False

        timings = result.get("timings", {})
        snapshot = SessionRestoreSnapshot(
            conversation_id=state.conversation_id,
            model_id=state.model_id,
            slot_id=state.slot_id,
            estimated_prompt_tokens=state.estimated_prompt_tokens,
            prefix_digest=state.prefix_digest,
            request_shape_digest=state.request_shape_digest,
            prompt_mode=state.prompt_mode,
            message_count=state.message_count,
            slot_message_count=state.slot_message_count,
            save_filename=filename,
            state_payload=self._serialize_continuation_state(state),
            saved_at=time.time(),
            last_access_at=time.time(),
            save_ms=timings.get("save_ms"),
            n_saved=int(result.get("n_saved", 0) or 0),
            n_written=int(result.get("n_written", 0) or 0),
            restore_count=0 if existing is None else existing.restore_count,
            last_restore_at=None if existing is None else existing.last_restore_at,
            last_restore_ms=None if existing is None else existing.last_restore_ms,
            last_restore_n_restored=(
                0 if existing is None else existing.last_restore_n_restored
            ),
            last_restore_n_read=0 if existing is None else existing.last_restore_n_read,
            last_restore_status="" if existing is None else existing.last_restore_status,
            last_restore_error="" if existing is None else existing.last_restore_error,
            runtime_signature=self._runtime_state_signature,
        )
        self._session_restore_store.put(snapshot)
        with self._lock:
            self._session_restore_counts["saved"] = (
                self._session_restore_counts.get("saved", 0) + 1
            )
            self._last_slot_save = {
                "ok": True,
                "conversation_key_hash": _hash_key(state.conversation_id),
                "slot_id": state.slot_id,
                "filename": filename,
                "save_ms": timings.get("save_ms"),
                "n_saved": result.get("n_saved"),
                "n_written": result.get("n_written"),
                "timestamp": round(time.time(), 3),
            }
        return True

    def _try_restore_session_snapshot(
        self,
        *,
        conversation_id: str,
        model_id: str,
        slot_id: int,
    ) -> bool:
        if not self._session_restore_enabled():
            return False
        snapshot = self._session_restore_store.get(
            conversation_id=conversation_id,
            model_id=model_id,
            touch=False,
        )
        if snapshot is None:
            with self._lock:
                self._session_restore_counts["restore_miss"] = (
                    self._session_restore_counts.get("restore_miss", 0) + 1
                )
                self._last_slot_restore = {
                    "ok": False,
                    "conversation_key_hash": _hash_key(conversation_id),
                    "slot_id": slot_id,
                    "filename": "",
                    "status": "missing_snapshot",
                    "timestamp": round(time.time(), 3),
                }
            return False

        slot_save_file = self._slot_save_path / snapshot.save_filename
        if not slot_save_file.exists():
            snapshot.last_restore_at = time.time()
            snapshot.last_restore_status = "missing_slot_save_file"
            snapshot.last_restore_error = str(slot_save_file)
            self._session_restore_store.put(snapshot)
            with self._lock:
                self._session_restore_counts["restore_error"] = (
                    self._session_restore_counts.get("restore_error", 0) + 1
                )
                self._last_slot_restore = {
                    "ok": False,
                    "conversation_key_hash": _hash_key(conversation_id),
                    "slot_id": slot_id,
                    "filename": snapshot.save_filename,
                    "status": snapshot.last_restore_status,
                    "error": snapshot.last_restore_error,
                    "timestamp": round(time.time(), 3),
                }
            return False

        try:
            result = self._slot_action(
                slot_id=slot_id,
                action="restore",
                payload={"filename": snapshot.save_filename},
            )
        except Exception as exc:
            snapshot.last_restore_at = time.time()
            snapshot.last_restore_status = "restore_error"
            snapshot.last_restore_error = str(exc)
            self._session_restore_store.put(snapshot)
            with self._lock:
                self._session_restore_counts["restore_error"] = (
                    self._session_restore_counts.get("restore_error", 0) + 1
                )
                self._last_slot_restore = {
                    "ok": False,
                    "conversation_key_hash": _hash_key(conversation_id),
                    "slot_id": slot_id,
                    "filename": snapshot.save_filename,
                    "status": snapshot.last_restore_status,
                    "error": snapshot.last_restore_error,
                    "timestamp": round(time.time(), 3),
                }
            return False

        timings = result.get("timings", {})
        snapshot.slot_id = slot_id
        snapshot.last_access_at = time.time()
        snapshot.restore_count += 1
        snapshot.last_restore_at = time.time()
        snapshot.last_restore_ms = timings.get("restore_ms")
        snapshot.last_restore_n_restored = int(result.get("n_restored", 0) or 0)
        snapshot.last_restore_n_read = int(result.get("n_read", 0) or 0)
        snapshot.last_restore_status = "restored"
        snapshot.last_restore_error = ""
        snapshot.runtime_signature = self._runtime_state_signature
        self._session_restore_store.put(snapshot)
        with self._lock:
            self._session_restore_counts["restored"] = (
                self._session_restore_counts.get("restored", 0) + 1
            )
            self._last_slot_restore = {
                "ok": True,
                "conversation_key_hash": _hash_key(conversation_id),
                "slot_id": slot_id,
                "filename": snapshot.save_filename,
                "restore_ms": timings.get("restore_ms"),
                "n_restored": result.get("n_restored"),
                "n_read": result.get("n_read"),
                "status": snapshot.last_restore_status,
                "timestamp": round(time.time(), 3),
            }
        return True

    def _persist_runtime_state_locked(self) -> None:
        single_slot_state = self._single_slot_continuation
        if single_slot_state is not None and self._single_slot_continuation_dirty:
            durable_snapshot = self._session_restore_store.get(
                conversation_id=single_slot_state.conversation_id,
                model_id=single_slot_state.model_id,
                touch=False,
            )
            if isinstance(getattr(durable_snapshot, "state_payload", None), dict):
                restored_state = self._deserialize_continuation_state(
                    durable_snapshot.state_payload
                )
                if restored_state is not None:
                    single_slot_state = restored_state
        bindings = []
        for session_key, binding in self._session_bindings.items():
            bindings.append(
                {
                    "session_key": session_key,
                    "binding": binding.to_dict(),
                }
            )
        payload = {
            "version": _PERSISTED_RUNTIME_STATE_VERSION,
            "updated_at": round(time.time(), 3),
            "runtime_signature": self._runtime_state_signature,
            "single_slot_recovery_pending": self._single_slot_recovery_pending,
            "single_slot_continuation": (
                None
                if single_slot_state is None
                else self._serialize_continuation_state(single_slot_state)
            ),
            "session_bindings": bindings,
        }
        self._runtime_state_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self._runtime_state_path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(payload, sort_keys=True, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(self._runtime_state_path)

    def _load_runtime_state(self) -> None:
        if not self._runtime_state_path.exists():
            return
        try:
            payload = json.loads(self._runtime_state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return

        stored_signature = str(payload.get("runtime_signature", "") or "")
        current_signature = self._current_runtime_signature()
        signature_matches = bool(
            stored_signature and current_signature and stored_signature == current_signature
        )
        self._runtime_state_signature = current_signature or stored_signature

        continuation_payload = payload.get("single_slot_continuation")
        if isinstance(continuation_payload, dict):
            state = self._deserialize_continuation_state(continuation_payload)
            if state is not None:
                self._single_slot_continuation = state
                self._single_slot_continuation_dirty = False
                self._single_slot_restored_from_disk = True
                self._single_slot_recovery_pending = not signature_matches

        if signature_matches:
            restored_bindings = OrderedDict()
            restored_owners: Dict[int, str] = {}
            for item in payload.get("session_bindings", []):
                if not isinstance(item, dict):
                    continue
                session_key = item.get("session_key")
                binding_payload = item.get("binding")
                if not session_key or not isinstance(binding_payload, dict):
                    continue
                try:
                    binding = SessionBinding(
                        slot_id=int(binding_payload["slot_id"]),
                        estimated_prompt_tokens=int(
                            binding_payload.get("estimated_prompt_tokens", 0)
                        ),
                        last_used_at=float(binding_payload.get("last_used_at", 0.0)),
                        sticky_key_hash=_hash_key(str(session_key)),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                restored_bindings[str(session_key)] = binding
                restored_owners[binding.slot_id] = str(session_key)
            self._session_bindings = restored_bindings
            self._slot_owners = restored_owners
            self._evict_oldest_binding_locked()
        else:
            self._session_bindings.clear()
            self._slot_owners.clear()

        with self._lock:
            self._persist_runtime_state_locked()

    def _evict_oldest_binding_locked(self) -> None:
        while len(self._session_bindings) > self.config.sticky_max_sessions:
            session_key, binding = self._session_bindings.popitem(last=False)
            owner = self._slot_owners.get(binding.slot_id)
            if owner == session_key:
                self._slot_owners.pop(binding.slot_id, None)

    def _bind_session_locked(
        self,
        session_key: str,
        *,
        slot_id: int,
        estimated_prompt_tokens: int,
    ) -> None:
        existing = self._session_bindings.pop(session_key, None)
        if existing and self._slot_owners.get(existing.slot_id) == session_key:
            self._slot_owners.pop(existing.slot_id, None)
        previous_owner = self._slot_owners.get(slot_id)
        if previous_owner and previous_owner != session_key:
            self._session_bindings.pop(previous_owner, None)
        self._slot_owners[slot_id] = session_key
        self._session_bindings[session_key] = SessionBinding(
            slot_id=slot_id,
            estimated_prompt_tokens=estimated_prompt_tokens,
            last_used_at=time.time(),
            sticky_key_hash=_hash_key(session_key),
        )
        self._evict_oldest_binding_locked()
        current_signature = self._current_runtime_signature()
        if current_signature:
            self._runtime_state_signature = current_signature
        self._persist_runtime_state_locked()

    def _binding_for_slot_locked(
        self,
        slot_id: int,
    ) -> tuple[Optional[str], Optional[SessionBinding]]:
        owner = self._slot_owners.get(slot_id)
        if not owner:
            return None, None
        binding = self._session_bindings.get(owner)
        if binding is None or binding.slot_id != slot_id:
            self._slot_owners.pop(slot_id, None)
            return None, None
        return owner, binding

    def _release_slot_owner_locked(self, slot_id: int) -> None:
        owner, binding = self._binding_for_slot_locked(slot_id)
        self._slot_owners.pop(slot_id, None)
        if owner and binding and self._session_bindings.get(owner) is binding:
            self._session_bindings.pop(owner, None)
        self._persist_runtime_state_locked()

    def _sync_runtime_state_locked(self) -> None:
        if not self._runtime_state_signature:
            return
        current_signature = self._current_runtime_signature()
        if current_signature == self._runtime_state_signature:
            return
        self._session_bindings.clear()
        self._slot_owners.clear()
        if self._single_slot_continuation is not None:
            self._single_slot_recovery_pending = True
        self._runtime_state_signature = current_signature
        self._persist_runtime_state_locked()

    def _record_continuation_decision(self, decision: ContinuationDecision) -> None:
        with self._lock:
            self._last_continuation_decision = decision
            self._recent_continuation_decisions.append(decision)
            self._continuation_counts[decision.reason] = self._continuation_counts.get(
                decision.reason, 0
            ) + 1

    def _single_slot_summary(self) -> Dict[str, Any]:
        with self._lock:
            state = None if self._single_slot_continuation is None else self._single_slot_continuation.to_dict()
            recent = [decision.to_dict() for decision in self._recent_continuation_decisions]
            last = (
                None
                if self._last_continuation_decision is None
                else self._last_continuation_decision.to_dict()
            )
            counts = dict(self._continuation_counts)
            recovery_pending = self._single_slot_recovery_pending
            restored_from_disk = self._single_slot_restored_from_disk
            durable_state_dirty = self._single_slot_continuation_dirty
            runtime_signature = self._runtime_state_signature
        return {
            "enabled": self.config.enable_session_stickiness and self.config.parallel_slots <= 1,
            "ttl_seconds": int(_SINGLE_SLOT_CONTINUATION_TTL_SECONDS),
            "recovery_pending": recovery_pending,
            "restored_from_disk": restored_from_disk,
            "durable_state_dirty": durable_state_dirty,
            "state_path": str(self._runtime_state_path),
            "runtime_signature": runtime_signature,
            "state": state,
            "counts": counts,
            "last_decision": last,
            "recent_decisions": recent,
        }

    def _session_restore_summary(self) -> Dict[str, Any]:
        snapshots = self._session_restore_store.list_snapshots(
            limit=_SESSION_RESTORE_RECENT_LIMIT,
        )
        public_snapshots = []
        restorable_snapshots = 0
        for snapshot in snapshots:
            payload = snapshot.to_public_dict()
            slot_save_file = self._slot_save_path / snapshot.save_filename
            payload["slot_save_exists"] = slot_save_file.exists()
            if payload["slot_save_exists"]:
                restorable_snapshots += 1
            public_snapshots.append(payload)
        with self._lock:
            counts = dict(self._session_restore_counts)
            last_save = None if self._last_slot_save is None else dict(self._last_slot_save)
            last_restore = (
                None if self._last_slot_restore is None else dict(self._last_slot_restore)
            )
        stats = self._session_restore_store.stats()
        stats["restorable_snapshots"] = restorable_snapshots
        return {
            "enabled": self._session_restore_enabled(),
            "slot_save_path": str(self._slot_save_path),
            "min_prompt_tokens": self._session_restore_threshold(),
            "stats": stats,
            "counts": counts,
            "last_save": last_save,
            "last_restore": last_restore,
            "snapshots": public_snapshots,
        }

    def _persist_session_snapshot_from_payload(
        self,
        *,
        path: str,
        payload: Dict[str, Any],
        response_payload: Optional[Dict[str, Any]],
    ) -> bool:
        if path not in {"v1/chat/completions", "v1/completions"}:
            return False
        conversation_id = _extract_routing_key(payload)
        slot_id = payload.get("id_slot")
        if not conversation_id or not isinstance(slot_id, int):
            return False
        if payload.get("cache_prompt") is False:
            return False
        estimated_prompt_tokens = _estimate_prompt_tokens(payload)
        effective_prompt_tokens = estimated_prompt_tokens
        raw_effective_prompt_tokens = payload.get("_omlx_effective_prompt_tokens")
        if isinstance(raw_effective_prompt_tokens, (int, float)):
            effective_prompt_tokens = max(0, int(raw_effective_prompt_tokens))
        next_state = self._build_continuation_state(
            path,
            payload,
            conversation_id,
            estimated_prompt_tokens,
        )
        next_state.slot_id = slot_id
        if (
            self.config.parallel_slots <= 1
            and effective_prompt_tokens < self.config.sticky_session_prompt_threshold
            and self._session_restore_store.get(
                conversation_id=conversation_id,
                model_id=next_state.model_id,
                touch=False,
            )
            is not None
        ):
            return False
        if (
            path == "v1/chat/completions"
            and next_state.prompt_mode == "messages"
            and isinstance(next_state.prompt_value, tuple)
            and isinstance(response_payload, dict)
        ):
            assistant_message = _response_assistant_message(response_payload)
            if assistant_message is not None:
                next_state.slot_prompt_value = next_state.prompt_value + (
                    _canonical_message(assistant_message),
                )
                next_state.slot_message_count = next_state.message_count + 1
        return self._persist_session_snapshot(next_state)

    def _flush_dirty_single_slot_snapshot(self) -> None:
        with self._lock:
            state = self._single_slot_continuation
            dirty = self._single_slot_continuation_dirty
        if not dirty or state is None:
            return
        fingerprint = (
            state.conversation_id,
            state.model_id,
            state.prefix_digest,
            state.request_shape_digest,
            state.slot_message_count,
        )
        if not self._persist_session_snapshot(state):
            return
        with self._lock:
            current = self._single_slot_continuation
            if current is None:
                return
            current_fingerprint = (
                current.conversation_id,
                current.model_id,
                current.prefix_digest,
                current.request_shape_digest,
                current.slot_message_count,
            )
            if current_fingerprint != fingerprint:
                return
            self._single_slot_continuation_dirty = False
            current_signature = self._current_runtime_signature()
            if current_signature:
                self._runtime_state_signature = current_signature
            self._persist_runtime_state_locked()

    def _build_continuation_state(
        self,
        path: str,
        payload: Dict[str, Any],
        conversation_id: str,
        estimated_prompt_tokens: int,
    ) -> ContinuationState:
        canonical_messages = _canonical_messages(payload)
        prompt_mode = "messages" if canonical_messages else "prompt"
        prompt_value: str | tuple[str, ...]
        message_count = 0
        if canonical_messages:
            prompt_value = canonical_messages
            message_count = len(canonical_messages)
            prefix_bytes = "\n".join(canonical_messages).encode("utf-8")
        else:
            prompt_value = _canonical_prompt(payload)
            prefix_bytes = str(prompt_value).encode("utf-8")
        prefix_digest = hashlib.sha256(prefix_bytes).hexdigest()[:16]
        return ContinuationState(
            conversation_id=conversation_id,
            slot_id=0,
            model_id=str(payload.get("model", "")),
            estimated_prompt_tokens=estimated_prompt_tokens,
            last_used_at=time.time(),
            prefix_digest=prefix_digest,
            request_shape_digest=_continuation_shape_digest(path, payload),
            prompt_mode=prompt_mode,
            prompt_value=prompt_value,
            message_count=message_count,
            slot_prompt_value=prompt_value,
            slot_message_count=message_count,
        )

    def _record_single_slot_outcome(
        self,
        prepared: Dict[str, Any],
        *,
        path: str,
        continuation_reason: str,
        continuation_hit: bool,
        prefix_drift: bool,
        suffix_only: bool,
    ) -> None:
        routing_key = _extract_routing_key(prepared) or ""
        estimated_prompt_tokens = _estimate_prompt_tokens(prepared)
        decision = ContinuationDecision(
            reason=continuation_reason,
            conversation_key_hash=_hash_key(routing_key) if routing_key else "",
            slot_id=int(prepared.get("id_slot", 0) or 0),
            estimated_prompt_tokens=estimated_prompt_tokens,
            continuation_hit=continuation_hit,
            prefix_drift=prefix_drift,
            suffix_only=suffix_only,
            timestamp=time.time(),
        )
        self._record_continuation_decision(decision)
        slot_reason_map = {
            "cold_start": "single_slot_cold_start",
            "hit": "single_slot_hit",
            "recovery_replay": "single_slot_recovery_replay",
            "prefix_drift": "single_slot_prefix_drift",
            "shape_changed": "single_slot_shape_changed",
            "model_changed": "single_slot_model_changed",
            "expired": "single_slot_expired",
            "slot_reset": "single_slot_slot_reset",
            "no_conversation_id": "single_slot_no_conversation_id",
            "disabled": "pass_through",
        }
        self._record_slot_decision(
            SlotRouteDecision(
                reason=slot_reason_map.get(continuation_reason, "pass_through"),
                path=path,
                slot_id=prepared.get("id_slot"),
                estimated_prompt_tokens=estimated_prompt_tokens,
                sticky_key_hash=_hash_key(routing_key) if routing_key else "",
                cache_prompt=bool(prepared.get("cache_prompt", True)),
                explicit_slot="id_slot" in prepared,
                timestamp=time.time(),
            )
        )

    def _prepare_single_slot_payload(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(payload)
        if prepared.get("cache_prompt") is None:
            prepared["cache_prompt"] = True
        routing_key = _extract_routing_key(prepared)
        if not self.config.enable_session_stickiness:
            self._record_single_slot_outcome(
                prepared,
                path=path,
                continuation_reason="disabled",
                continuation_hit=False,
                prefix_drift=False,
                suffix_only=False,
            )
            return prepared
        if not routing_key:
            self._record_single_slot_outcome(
                prepared,
                path=path,
                continuation_reason="no_conversation_id",
                continuation_hit=False,
                prefix_drift=False,
                suffix_only=False,
            )
            return prepared

        prepared["id_slot"] = 0
        estimated_prompt_tokens = _estimate_prompt_tokens(prepared)
        next_state = self._build_continuation_state(
            path,
            prepared,
            routing_key,
            estimated_prompt_tokens,
        )

        reason = "cold_start"
        continuation_hit = False
        prefix_drift = False
        suffix_only = False
        restore_needed = False
        current: Optional[ContinuationState] = None
        with self._lock:
            self._sync_runtime_state_locked()
            current = self._single_slot_continuation
            if current is not None:
                age_seconds = time.time() - current.last_used_at
                if age_seconds > _SINGLE_SLOT_CONTINUATION_TTL_SECONDS:
                    reason = "expired"
                    self._single_slot_continuation = None
                    self._persist_runtime_state_locked()
                elif current.slot_id != 0:
                    reason = "slot_reset"
                    self._single_slot_continuation = None
                    self._persist_runtime_state_locked()
                elif current.conversation_id != routing_key:
                    reason = "cold_start"
                elif current.model_id != next_state.model_id:
                    reason = "model_changed"
                elif current.request_shape_digest != next_state.request_shape_digest:
                    reason = "shape_changed"
                elif current.prompt_mode != next_state.prompt_mode:
                    reason = "shape_changed"
                elif current.prompt_mode == "messages":
                    prompt_value = next_state.prompt_value
                    current_prompt_value = current.prompt_value
                    if not (
                        isinstance(prompt_value, tuple)
                        and isinstance(current_prompt_value, tuple)
                        and len(prompt_value) >= len(current_prompt_value)
                        and prompt_value[: len(current_prompt_value)] == current_prompt_value
                    ):
                        reason = "prefix_drift"
                        prefix_drift = True
                    elif self._single_slot_recovery_pending:
                        restore_needed = True
                    else:
                        reason = "hit"
                        continuation_hit = True
                else:
                    prompt_value = str(next_state.prompt_value)
                    current_prompt_value = str(current.prompt_value)
                    if not prompt_value.startswith(current_prompt_value):
                        reason = "prefix_drift"
                        prefix_drift = True
                    elif self._single_slot_recovery_pending:
                        restore_needed = True
                    else:
                        reason = "hit"
                        continuation_hit = True

        if restore_needed:
            if self._try_restore_session_snapshot(
                conversation_id=routing_key,
                model_id=next_state.model_id,
                slot_id=0,
            ):
                with self._lock:
                    self._single_slot_recovery_pending = False
                    self._single_slot_restored_from_disk = True
                    self._persist_runtime_state_locked()
                reason = "hit"
                continuation_hit = True
            else:
                with self._lock:
                    self._session_restore_counts["restore_fallback"] = (
                        self._session_restore_counts.get("restore_fallback", 0) + 1
                    )
                reason = "recovery_replay"

        if continuation_hit and next_state.prompt_mode == "messages":
            messages = prepared.get("messages")
            slot_prompt_value = None if current is None else current.slot_prompt_value
            slot_message_count = 0 if current is None else current.slot_message_count
            if (
                isinstance(messages, list)
                and isinstance(next_state.prompt_value, tuple)
                and isinstance(slot_prompt_value, tuple)
                and slot_message_count > 0
                and len(messages) >= slot_message_count
                and len(next_state.prompt_value) >= len(slot_prompt_value)
                and next_state.prompt_value[: len(slot_prompt_value)] == slot_prompt_value
            ):
                suffix_messages = messages[slot_message_count:]
                if suffix_messages:
                    anchor_count = 1
                    if slot_message_count > 0 and len(slot_prompt_value) >= slot_message_count:
                        last_slot_role = _canonical_message_role(
                            slot_prompt_value[slot_message_count - 1]
                        )
                        if last_slot_role != "assistant":
                            anchor_count = 1 if slot_message_count <= 2 else 2
                    start_index = max(0, slot_message_count - anchor_count)
                    prepared["messages"] = messages[start_index:]
                    suffix_only = True

        self._record_single_slot_outcome(
            prepared,
            path=path,
            continuation_reason=reason,
            continuation_hit=continuation_hit,
            prefix_drift=prefix_drift,
            suffix_only=suffix_only,
        )
        return prepared

    def _finalize_single_slot_request(
        self,
        path: str,
        payload: Dict[str, Any],
        response_ok: bool,
        *,
        response_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if path not in {"v1/chat/completions", "v1/completions"} or not response_ok:
            return
        persisted_snapshot = self._persist_session_snapshot_from_payload(
            path=path,
            payload=payload,
            response_payload=response_payload,
        )
        if self.config.parallel_slots > 1:
            return
        conversation_id = _extract_routing_key(payload)
        if not conversation_id or payload.get("id_slot") != 0:
            return
        estimated_prompt_tokens = _estimate_prompt_tokens(payload)
        next_state = self._build_continuation_state(
            path,
            payload,
            conversation_id,
            estimated_prompt_tokens,
        )
        if (
            path == "v1/chat/completions"
            and next_state.prompt_mode == "messages"
            and isinstance(next_state.prompt_value, tuple)
            and isinstance(response_payload, dict)
        ):
            assistant_message = _response_assistant_message(response_payload)
            if assistant_message is not None:
                next_state.slot_prompt_value = next_state.prompt_value + (
                    _canonical_message(assistant_message),
                )
                next_state.slot_message_count = next_state.message_count + 1
        with self._lock:
            self._runtime_state_signature = self._current_runtime_signature()
            self._single_slot_continuation = next_state
            self._single_slot_continuation_dirty = not persisted_snapshot
            self._single_slot_recovery_pending = False
            self._single_slot_restored_from_disk = False
            if persisted_snapshot:
                self._persist_runtime_state_locked()

    def _record_slot_decision(self, decision: SlotRouteDecision) -> None:
        with self._lock:
            self._last_slot_decision = decision
            self._recent_slot_decisions.append(decision)
            self._slot_route_counts[decision.reason] = self._slot_route_counts.get(
                decision.reason, 0
            ) + 1

    def _choose_idle_slot(
        self,
        slots_payload: Dict[str, Any],
    ) -> Optional[int]:
        slots = self._extract_slots_list(slots_payload)
        candidate_ids: List[int] = []
        fallback_ids: List[tuple[float, int, int]] = []
        for slot in slots:
            slot_id = slot.get("id")
            if not isinstance(slot_id, int):
                continue
            if slot.get("is_processing"):
                continue
            with self._lock:
                _, binding = self._binding_for_slot_locked(slot_id)
            if binding is None:
                candidate_ids.append(slot_id)
            else:
                fallback_ids.append(
                    (
                        binding.last_used_at,
                        binding.estimated_prompt_tokens,
                        slot_id,
                    )
                )
        if candidate_ids:
            return sorted(candidate_ids)[0]
        if fallback_ids:
            fallback_ids.sort()
            return fallback_ids[0][2]
        return None

    def _choose_idle_unowned_slot(self, slots_payload: Dict[str, Any]) -> Optional[int]:
        slots = self._extract_slots_list(slots_payload)
        candidate_ids: List[int] = []
        for slot in slots:
            slot_id = slot.get("id")
            if not isinstance(slot_id, int):
                continue
            if slot.get("is_processing"):
                continue
            if self._slot_owners.get(slot_id) is None:
                candidate_ids.append(slot_id)
        if not candidate_ids:
            return None
        return sorted(candidate_ids)[0]

    def _choose_recyclable_owned_slot(self, slots_payload: Dict[str, Any]) -> Optional[int]:
        slots = self._extract_slots_list(slots_payload)
        now = time.time()
        candidates: List[tuple[float, int, int]] = []
        with self._lock:
            for slot in slots:
                slot_id = slot.get("id")
                if not isinstance(slot_id, int):
                    continue
                if slot.get("is_processing"):
                    continue
                _, binding = self._binding_for_slot_locked(slot_id)
                if binding is None:
                    continue
                age_seconds = now - binding.last_used_at
                if age_seconds < _RECYCLE_OWNED_IDLE_SLOT_MIN_AGE_SECONDS:
                    continue
                candidates.append(
                    (
                        binding.last_used_at,
                        binding.estimated_prompt_tokens,
                        slot_id,
                    )
                )
        if not candidates:
            return None
        candidates.sort()
        return candidates[0][2]

    def _maybe_restore_session_binding(
        self,
        *,
        prepared: Dict[str, Any],
        routing_key: str,
        estimated_prompt_tokens: int,
    ) -> Optional[int]:
        if not self._session_restore_enabled():
            return None
        model_id = str(prepared.get("model", "") or "")
        if not model_id:
            return None
        snapshot = self._session_restore_store.get(
            conversation_id=routing_key,
            model_id=model_id,
            touch=False,
        )
        if snapshot is None:
            return None
        slot_id = max(0, min(snapshot.slot_id, self.config.parallel_slots - 1))
        if self._try_restore_session_snapshot(
            conversation_id=routing_key,
            model_id=model_id,
            slot_id=slot_id,
        ):
            with self._lock:
                self._bind_session_locked(
                    routing_key,
                    slot_id=slot_id,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                )
            prepared["id_slot"] = slot_id
            return slot_id
        with self._lock:
            self._session_restore_counts["restore_fallback"] = (
                self._session_restore_counts.get("restore_fallback", 0) + 1
            )
        return None

    def _slot_router_summary(self, slots_payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._sync_runtime_state_locked()
            bindings = [binding.to_dict() for binding in self._session_bindings.values()]
            recent = [decision.to_dict() for decision in self._recent_slot_decisions]
            last_decision = (
                None if self._last_slot_decision is None else self._last_slot_decision.to_dict()
            )
            counts = dict(self._slot_route_counts)
        return {
            "enabled": self.config.enable_session_stickiness,
            "sticky_session_prompt_threshold": self.config.sticky_session_prompt_threshold,
            "sticky_max_sessions": self.config.sticky_max_sessions,
            "bindings": bindings,
            "counts": counts,
            "last_decision": last_decision,
            "recent_decisions": recent,
            "slot_summary": self._slot_summary(slots_payload),
        }

    def _prepare_proxy_payload(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prepared = _merge_extra_body(dict(payload))
        if path not in {"v1/chat/completions", "v1/completions"}:
            return prepared
        if prepared.get("cache_prompt") is None:
            prepared["cache_prompt"] = True
        if prepared.get("think") is False:
            chat_template_kwargs = prepared.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            else:
                chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs.setdefault("enable_thinking", False)
            prepared["chat_template_kwargs"] = chat_template_kwargs
        if path == "v1/chat/completions" and _disable_thinking_requested(prepared):
            chat_template_kwargs = prepared.get("chat_template_kwargs")
            if not isinstance(chat_template_kwargs, dict):
                chat_template_kwargs = {}
            else:
                chat_template_kwargs = dict(chat_template_kwargs)
            chat_template_kwargs["enable_thinking"] = False
            prepared["chat_template_kwargs"] = chat_template_kwargs
            prepared.setdefault("enableThinking", False)
            prepared.setdefault("reasoning", False)
            prepared.setdefault("reasoning_budget", 0)
            prepared.setdefault("reasoning_format", "none")
            prepared.setdefault("thinking_forced_open", False)
            prepared.pop("think", None)
        if self.config.parallel_slots <= 1:
            return self._prepare_single_slot_payload(path, prepared)
        if not self.config.enable_session_stickiness:
            self._record_slot_decision(
                SlotRouteDecision(
                    reason="pass_through",
                    path=path,
                    slot_id=prepared.get("id_slot"),
                    estimated_prompt_tokens=_estimate_prompt_tokens(prepared),
                    sticky_key_hash="",
                    cache_prompt=bool(prepared.get("cache_prompt", True)),
                    explicit_slot="id_slot" in prepared,
                    timestamp=time.time(),
                )
            )
            return prepared

        estimated_prompt_tokens = _estimate_prompt_tokens(prepared)
        routing_key = _extract_routing_key(prepared)
        sticky_key_hash = _hash_key(routing_key) if routing_key else ""

        if "id_slot" in prepared:
            if routing_key and isinstance(prepared["id_slot"], int):
                with self._lock:
                    self._bind_session_locked(
                        routing_key,
                        slot_id=prepared["id_slot"],
                        estimated_prompt_tokens=estimated_prompt_tokens,
                    )
            self._record_slot_decision(
                SlotRouteDecision(
                    reason="explicit_slot",
                    path=path,
                    slot_id=prepared.get("id_slot"),
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    sticky_key_hash=sticky_key_hash,
                    cache_prompt=bool(prepared.get("cache_prompt", True)),
                    explicit_slot=True,
                    timestamp=time.time(),
                )
            )
            return prepared

        if not routing_key:
            reason = "no_routing_key"
            chosen_slot = None
            if self._slot_owners:
                slots_payload = self._request_optional_json("slots")
                chosen_slot = self._choose_idle_unowned_slot(slots_payload)
                if chosen_slot is None:
                    chosen_slot = self._choose_recyclable_owned_slot(slots_payload)
                    if chosen_slot is not None:
                        with self._lock:
                            self._release_slot_owner_locked(chosen_slot)
                        prepared["id_slot"] = chosen_slot
                        reason = "unkeyed_recycled_slot"
            if chosen_slot is not None:
                if "id_slot" not in prepared:
                    prepared["id_slot"] = chosen_slot
                if reason == "no_routing_key":
                    reason = "unkeyed_idle_slot"
            elif estimated_prompt_tokens < self.config.sticky_session_prompt_threshold:
                reason = "short_prompt"
            self._record_slot_decision(
                SlotRouteDecision(
                    reason=reason,
                    path=path,
                    slot_id=prepared.get("id_slot"),
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    sticky_key_hash="",
                    cache_prompt=bool(prepared.get("cache_prompt", True)),
                    explicit_slot=False,
                    timestamp=time.time(),
                )
            )
            return prepared

        with self._lock:
            self._sync_runtime_state_locked()
            binding = self._session_bindings.get(routing_key)
            if binding and binding.slot_id >= self.config.parallel_slots:
                self._session_bindings.pop(routing_key, None)
                self._slot_owners.pop(binding.slot_id, None)
                self._persist_runtime_state_locked()
                binding = None
            if binding:
                binding.last_used_at = time.time()
                binding.estimated_prompt_tokens = estimated_prompt_tokens
                self._session_bindings.move_to_end(routing_key)
                current_signature = self._current_runtime_signature()
                if current_signature:
                    self._runtime_state_signature = current_signature
                self._persist_runtime_state_locked()
                prepared["id_slot"] = binding.slot_id
                self._record_slot_decision(
                    SlotRouteDecision(
                        reason="sticky_existing",
                        path=path,
                        slot_id=binding.slot_id,
                        estimated_prompt_tokens=estimated_prompt_tokens,
                        sticky_key_hash=sticky_key_hash,
                        cache_prompt=bool(prepared.get("cache_prompt", True)),
                        explicit_slot=False,
                        timestamp=time.time(),
                    )
                )
                return prepared

        restored_slot = self._maybe_restore_session_binding(
            prepared=prepared,
            routing_key=routing_key,
            estimated_prompt_tokens=estimated_prompt_tokens,
        )
        if restored_slot is not None:
            self._record_slot_decision(
                SlotRouteDecision(
                    reason="sticky_restored",
                    path=path,
                    slot_id=restored_slot,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    sticky_key_hash=sticky_key_hash,
                    cache_prompt=bool(prepared.get("cache_prompt", True)),
                    explicit_slot=False,
                    timestamp=time.time(),
                )
            )
            return prepared

        if estimated_prompt_tokens < self.config.sticky_session_prompt_threshold:
            chosen_slot = None
            recycled_slot = False
            if self._slot_owners:
                slots_payload = self._request_optional_json("slots")
                chosen_slot = self._choose_idle_unowned_slot(slots_payload)
                if chosen_slot is None:
                    chosen_slot = self._choose_recyclable_owned_slot(slots_payload)
                    if chosen_slot is not None:
                        recycled_slot = True
                        with self._lock:
                            self._release_slot_owner_locked(chosen_slot)
            reason = "short_prompt"
            if chosen_slot is not None:
                prepared["id_slot"] = chosen_slot
                reason = (
                    "short_prompt_recycled_slot"
                    if recycled_slot
                    else "short_prompt_unowned_slot"
                )
            self._record_slot_decision(
                SlotRouteDecision(
                    reason=reason,
                    path=path,
                    slot_id=prepared.get("id_slot"),
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    sticky_key_hash=sticky_key_hash,
                    cache_prompt=bool(prepared.get("cache_prompt", True)),
                    explicit_slot=False,
                    timestamp=time.time(),
                )
            )
            return prepared

        slots_payload = self._request_optional_json("slots")
        chosen_slot = self._choose_idle_slot(slots_payload)
        if chosen_slot is None:
            self._record_slot_decision(
                SlotRouteDecision(
                    reason="sticky_no_idle_slot",
                    path=path,
                    slot_id=None,
                    estimated_prompt_tokens=estimated_prompt_tokens,
                    sticky_key_hash=sticky_key_hash,
                    cache_prompt=bool(prepared.get("cache_prompt", True)),
                    explicit_slot=False,
                    timestamp=time.time(),
                )
            )
            return prepared

        with self._lock:
            self._bind_session_locked(
                routing_key,
                slot_id=chosen_slot,
                estimated_prompt_tokens=estimated_prompt_tokens,
            )
        prepared["id_slot"] = chosen_slot
        self._record_slot_decision(
            SlotRouteDecision(
                reason="sticky_new_long_prompt",
                path=path,
                slot_id=chosen_slot,
                estimated_prompt_tokens=estimated_prompt_tokens,
                sticky_key_hash=sticky_key_hash,
                cache_prompt=bool(prepared.get("cache_prompt", True)),
                explicit_slot=False,
                timestamp=time.time(),
            )
        )
        return prepared

    def _sanitize_disable_thinking_response(
        self,
        *,
        path: str,
        request_payload: Dict[str, Any],
        response_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        if path not in {"v1/chat/completions", "v1/completions"}:
            return response_payload
        if not _disable_thinking_requested(request_payload):
            return response_payload

        sanitized = deepcopy(response_payload)
        choices = sanitized.get("choices")
        if not isinstance(choices, list):
            return sanitized

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if path == "v1/chat/completions":
                message = choice.get("message")
                if not isinstance(message, dict):
                    continue
                content = message.get("content")
                if isinstance(content, str):
                    message["content"] = _strip_thinking_content(content)
            else:
                content = choice.get("text")
                if isinstance(content, str):
                    choice["text"] = _strip_thinking_content(content)
        return sanitized

    def _rewrite_json_response(
        self,
        response: Any,
        payload: Dict[str, Any],
    ) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if hasattr(response, "_json_data"):
            response._json_data = payload
        if hasattr(response, "_content"):
            response._content = encoded
        try:
            response.text = encoded.decode("utf-8")
        except Exception:
            pass
        headers = getattr(response, "headers", None)
        if isinstance(headers, dict):
            headers["content-length"] = str(len(encoded))

    def _request_optional_json(self, path: str, *, timeout: float = 10.0) -> Dict[str, Any]:
        try:
            response = self._request("GET", path, timeout=timeout)
        except Exception:
            return {}
        if not response.ok:
            return {}
        payload = _coerce_response_payload(response)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"data": payload}
        return {}

    def _metrics_excerpt(self, max_lines: int = 20) -> List[str]:
        try:
            response = self._request("GET", "metrics", timeout=10)
        except Exception:
            return []
        if not response.ok:
            return []
        return response.text.splitlines()[:max_lines]

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.health():
                return
            if not self.process_manager.is_running():
                logs = self.process_manager.logs(lines=40)["lines"]
                log_tail = "\n".join(logs)
                raise BackendError(
                    "llama.cpp runtime exited before becoming ready"
                    + (f":\n{log_tail}" if log_tail else "")
                )
            time.sleep(0.5)

        logs = self.process_manager.logs(lines=40)["lines"]
        log_tail = "\n".join(logs)
        raise BackendError(
            f"llama.cpp runtime did not become ready within {self.config.startup_timeout_seconds} seconds"
            + (f":\n{log_tail}" if log_tail else "")
        )

    def collect_metrics(self) -> RuntimeMetrics:
        metrics = super().collect_metrics()
        props = self._request_optional_json("props")
        slots = self._request_optional_json("slots")
        details: Dict[str, Any] = dict(metrics.details or {})
        details.update({
            "diagnostics": self.diagnostics().to_dict(),
            "continuation": self._single_slot_summary(),
            "session_restore": self._session_restore_summary(),
        })
        if props:
            details["props"] = props
        if slots:
            details["slots"] = slots
            details["slot_router"] = self._slot_router_summary(slots)
        metrics_excerpt = self._metrics_excerpt()
        if metrics_excerpt:
            details["metrics_excerpt"] = metrics_excerpt
        metrics.details = details
        return metrics

    def proxy(self, method: str, path: str, **kwargs: Any):
        model_capabilities = infer_model_capabilities(
            self.config.artifact_path,
            self.config.model_repo_id,
        )
        if path == "v1/rerank" and model_capabilities.rerank:
            path = "reranking"
        payload = kwargs.get("json")
        if isinstance(payload, dict):
            state_payload = deepcopy(payload)
            prepared = self._prepare_proxy_payload(path, payload)
            kwargs["json"] = prepared
            state_payload["_omlx_effective_prompt_tokens"] = _estimate_prompt_tokens(prepared)
            for key in (
                "cache_prompt",
                "id_slot",
                "chat_template_kwargs",
                "enableThinking",
                "reasoning",
                "reasoning_budget",
                "reasoning_format",
                "thinking_forced_open",
            ):
                if key in prepared:
                    state_payload[key] = deepcopy(prepared[key])
        else:
            prepared = None
            state_payload = None
        try:
            response = super().proxy(method, path, **kwargs)
        except BackendError:
            if isinstance(prepared, dict) and path in {"v1/chat/completions", "v1/completions"}:
                with self._lock:
                    if self._single_slot_continuation is not None:
                        self._single_slot_recovery_pending = True
                    self._persist_runtime_state_locked()
            raise
        if isinstance(prepared, dict):
            response_payload = None
            content_type = response.headers.get("content-type", "")
            if response.ok and "application/json" in content_type:
                try:
                    response_payload = response.json()
                    if isinstance(response_payload, dict):
                        response_payload = self._sanitize_disable_thinking_response(
                            path=path,
                            request_payload=prepared,
                            response_payload=response_payload,
                        )
                        self._rewrite_json_response(response, response_payload)
                except Exception:
                    response_payload = None
            elif path in {"v1/chat/completions", "v1/completions"}:
                with self._lock:
                    if self._single_slot_continuation is not None:
                        self._single_slot_recovery_pending = True
                    self._persist_runtime_state_locked()
            self._finalize_single_slot_request(
                path,
                state_payload if isinstance(state_payload, dict) else prepared,
                response.ok,
                response_payload=response_payload,
            )
        return response

    def start_runtime(self) -> Dict[str, Any]:
        command = self._build_launch_command()
        result = self.process_manager.start(command, env=os.environ.copy())
        if result.get("reason") == "already_running":
            return {
                **result,
                "mode": "llama_cpp",
                "command": self.process_manager.command(),
            }
        self._wait_until_ready()
        with self._lock:
            if self._single_slot_continuation is not None:
                self._single_slot_recovery_pending = True
            self._session_bindings.clear()
            self._slot_owners.clear()
            self._runtime_state_signature = self._current_runtime_signature()
            self._persist_runtime_state_locked()
        return {**result, "mode": "llama_cpp"}

    def stop_runtime(self) -> Dict[str, Any]:
        self._flush_dirty_single_slot_snapshot()
        result = self.process_manager.stop()
        with self._lock:
            if self._single_slot_continuation is not None:
                self._single_slot_recovery_pending = True
            self._session_bindings.clear()
            self._slot_owners.clear()
            self._runtime_state_signature = self._current_runtime_signature()
            self._persist_runtime_state_locked()
        return {**result, "mode": "llama_cpp"}

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        return {**self.process_manager.logs(lines=lines), "mode": "llama_cpp"}

    def cache_report(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "continuation": self._single_slot_summary(),
            "session_restore": self._session_restore_summary(),
        }
        props = self._request_optional_json("props")
        slots = self._request_optional_json("slots")
        if props:
            payload["props"] = props
        if slots:
            payload["slots"] = slots
            payload["slot_router"] = self._slot_router_summary(slots)
        return payload

    def capabilities(self) -> BackendCapabilities:
        capabilities = infer_model_capabilities(
            self.config.artifact_path,
            self.config.model_repo_id,
        )
        is_rerank = capabilities.rerank
        has_mmproj = bool(self.config.mmproj_path)
        return BackendCapabilities(
            chat_completions=not is_rerank,
            completions=not is_rerank,
            embeddings=capabilities.embeddings,
            rerank=capabilities.rerank,
            vision_chat=capabilities.vision_chat or has_mmproj,
            ocr=capabilities.ocr or has_mmproj,
        )


@dataclass
class ModelPoolEvent:
    action: str
    model_id: str
    reason: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "model_id": self.model_id,
            "reason": self.reason,
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class _PooledModelHandle:
    registration: LlamaCppModelRegistration
    adapter: BackendAdapter
    loaded: bool = False
    loaded_at: Optional[float] = None
    last_used_at: Optional[float] = None
    last_load_reason: str = ""
    last_unload_reason: str = ""
    last_eviction_reason: str = ""
    request_count: int = 0


def _sanitize_model_dir_name(model_id: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-")
    return sanitized or _hash_key(model_id)


class LlamaCppModelPoolAdapter(BackendAdapter):
    """Pool wrapper that manages multiple llama.cpp GGUF runtimes."""

    def __init__(self, runtime_config: DGXRuntimeConfig, state_dir: Path) -> None:
        self.runtime_config = runtime_config
        self.config = runtime_config.backend
        self.base_url = self.config.base_url.rstrip("/")
        self.state_dir = Path(state_dir).expanduser().resolve() / "llama_cpp_model_pool"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._recent_events: Deque[ModelPoolEvent] = deque(maxlen=64)
        self._models: "OrderedDict[str, _PooledModelHandle]" = OrderedDict()
        self._default_model_id = self._resolve_default_model_id()

        for model_id, registration in self._resolved_registrations().items():
            self._register_model_locked(
                registration,
                is_default=(model_id == self._default_model_id),
                initial=True,
            )

    @classmethod
    def from_runtime_config(
        cls,
        runtime_config: DGXRuntimeConfig,
        state_dir: str | Path,
    ) -> "LlamaCppModelPoolAdapter":
        return cls(runtime_config, Path(state_dir))

    def _resolve_default_model_id(self) -> str:
        for model_id, profile in self.runtime_config.models.items():
            if profile.is_default:
                return model_id
        if self.runtime_config.models:
            return next(iter(self.runtime_config.models.keys()))
        if self.config.model_pool.models:
            return next(iter(self.config.model_pool.models.keys()))
        if self.config.model_repo_id:
            return self.config.model_repo_id
        if self.config.artifact_path:
            return Path(self.config.artifact_path).stem or "default"
        return "default"

    def _resolve_model_id(self, requested: Optional[str]) -> str:
        if requested:
            candidate = str(requested)
            if candidate in self._models:
                return candidate
            for model_id, handle in self._models.items():
                if handle.registration.model_alias and handle.registration.model_alias == candidate:
                    return model_id
            return candidate
        return self._default_model_id

    def _derive_model_base_url(self, ordinal: int) -> str:
        parsed = urlparse(self.config.base_url)
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 30000
        path = parsed.path.rstrip("/")
        suffix = f"{path}" if path else ""
        return f"{scheme}://{host}:{port + ordinal}{suffix}"

    def _resolved_registrations(self) -> "OrderedDict[str, LlamaCppModelRegistration]":
        resolved: "OrderedDict[str, LlamaCppModelRegistration]" = OrderedDict()
        explicit = self.config.model_pool.models

        primary_profile = self.runtime_config.models.get(self._default_model_id)
        if self._default_model_id not in explicit:
            primary_artifact_path = self.config.artifact_path
            resolved[self._default_model_id] = LlamaCppModelRegistration(
                model_id=self._default_model_id,
                model_alias=None if primary_profile is None else primary_profile.model_alias,
                backend_kind="llama_cpp",
                model_repo_id="" if primary_artifact_path else self.config.model_repo_id,
                artifact_path=primary_artifact_path,
                gguf_variant=self.config.gguf_variant,
                base_url=self.config.base_url,
                launcher_cmd=self.config.launcher_cmd,
                serving_preset=self.config.serving_preset,
                ctx_size=self.config.ctx_size,
                parallel_slots=self.config.parallel_slots,
                pinned=True,
                ttl_seconds=0,
                idle_unload_seconds=0,
            )

        for model_id, registration in explicit.items():
            profile = self.runtime_config.models.get(model_id)
            ordinal = 0 if model_id == self._default_model_id else len(resolved)
            use_parent_artifact = (
                not registration.artifact_path
                and not registration.model_repo_id
                and registration.backend_kind != "openai_compatible"
            )
            artifact_path = (
                self.config.artifact_path if use_parent_artifact else registration.artifact_path
            )
            resolved[model_id] = LlamaCppModelRegistration(
                model_id=model_id,
                model_alias=registration.model_alias
                or (None if profile is None else profile.model_alias),
                backend_kind=registration.backend_kind,
                backend_model_name=registration.backend_model_name,
                model_repo_id=registration.model_repo_id
                or ("" if artifact_path else self.config.model_repo_id),
                artifact_path=artifact_path,
                mmproj_path=registration.mmproj_path or self.config.mmproj_path,
                gguf_variant=registration.gguf_variant or self.config.gguf_variant,
                base_url=registration.base_url
                or (self.config.base_url if model_id == self._default_model_id else self._derive_model_base_url(ordinal)),
                launcher_cmd=registration.launcher_cmd,
                serving_preset=registration.serving_preset,
                ctx_size=registration.ctx_size,
                parallel_slots=registration.parallel_slots,
                pinned=registration.pinned,
                ttl_seconds=registration.ttl_seconds
                or self.config.model_pool.default_ttl_seconds,
                idle_unload_seconds=registration.idle_unload_seconds
                or self.config.model_pool.default_idle_unload_seconds,
            )
        return resolved

    def _adapter_config_for_registration(
        self,
        registration: LlamaCppModelRegistration,
    ) -> BackendConfig:
        use_parent_artifact = (
            not registration.artifact_path
            and not registration.model_repo_id
            and registration.backend_kind != "openai_compatible"
        )
        child_config = replace(
            self.config,
            base_url=registration.base_url or self.config.base_url,
            model_repo_id=registration.model_repo_id
            or ("" if registration.artifact_path else self.config.model_repo_id),
            artifact_path=(
                self.config.artifact_path if use_parent_artifact else registration.artifact_path
            ),
            mmproj_path=registration.mmproj_path or self.config.mmproj_path,
            gguf_variant=registration.gguf_variant or self.config.gguf_variant,
            launcher_cmd=registration.launcher_cmd,
            model_pool=self.config.model_pool,
        )
        if registration.serving_preset:
            apply_llama_cpp_serving_preset(child_config, registration.serving_preset)
        if registration.ctx_size > 0:
            child_config.ctx_size = registration.ctx_size
        if registration.parallel_slots > 0:
            child_config.parallel_slots = registration.parallel_slots
        return child_config

    def _record_event_locked(self, action: str, model_id: str, reason: str) -> None:
        self._recent_events.append(
            ModelPoolEvent(
                action=action,
                model_id=model_id,
                reason=reason,
                timestamp=time.time(),
            )
        )

    def _register_model_locked(
        self,
        registration: LlamaCppModelRegistration,
        *,
        is_default: bool,
        initial: bool,
    ) -> None:
        existing = self._models.get(registration.model_id)
        if existing is not None and existing.loaded:
            self._unload_model_locked(
                registration.model_id,
                reason="registration_updated",
            )

        model_state_dir = self.state_dir / _sanitize_model_dir_name(registration.model_id)
        if registration.backend_kind == "openai_compatible":
            external_model_name = (
                registration.backend_model_name
                or registration.model_repo_id
                or registration.model_alias
                or registration.model_id
            )
            adapter: BackendAdapter = ExternalOpenAIModelAdapter(
                base_url=registration.base_url or self.config.base_url,
                target_model_name=external_model_name,
                capabilities=BackendCapabilities(
                    chat_completions=bool(
                        registration.supports_vision or registration.supports_ocr
                    ),
                    completions=bool(
                        registration.supports_vision or registration.supports_ocr
                    ),
                    embeddings=registration.supports_embeddings,
                    rerank=registration.supports_rerank,
                    vision_chat=registration.supports_vision,
                    ocr=registration.supports_ocr,
                ),
            )
        else:
            adapter = LlamaCppBackendAdapter.from_backend_config(
                self._adapter_config_for_registration(registration),
                model_state_dir,
            )
        handle = _PooledModelHandle(registration=registration, adapter=adapter)
        if registration.backend_kind == "openai_compatible":
            now = time.time()
            handle.loaded = True
            handle.loaded_at = now
            handle.last_used_at = now
            handle.last_load_reason = "external_backend"
        elif isinstance(adapter, LlamaCppBackendAdapter) and adapter.process_manager.is_running():
            now = time.time()
            handle.loaded = True
            handle.loaded_at = now
            handle.last_used_at = now
            handle.last_load_reason = "discovered_running"
        self._models[registration.model_id] = handle
        if is_default:
            self._default_model_id = registration.model_id
        if not initial:
            self._record_event_locked("register", registration.model_id, "registered")

    def register_model(
        self,
        registration: LlamaCppModelRegistration,
        *,
        is_default: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            self._register_model_locked(
                registration,
                is_default=is_default,
                initial=False,
            )
            return self.model_pool_diagnostics()

    def _require_handle_locked(self, model_id: str) -> _PooledModelHandle:
        handle = self._models.get(model_id)
        if handle is None:
            raise BackendError(f"unknown llama.cpp pool model: {model_id}")
        return handle

    def _loaded_items_locked(self) -> List[tuple[str, _PooledModelHandle]]:
        return [
            (model_id, handle)
            for model_id, handle in self._models.items()
            if handle.loaded
        ]

    def _capacity_locked(self) -> int:
        return max(1, int(self.config.model_pool.max_loaded_models))

    def _unload_model_locked(self, model_id: str, *, reason: str) -> Dict[str, Any]:
        handle = self._require_handle_locked(model_id)
        if not handle.loaded:
            return {"model_id": model_id, "loaded": False, "reason": "not_loaded"}
        result = handle.adapter.stop_runtime()
        handle.loaded = False
        handle.loaded_at = None
        handle.last_unload_reason = reason
        if "eviction" in reason:
            handle.last_eviction_reason = reason
        self._record_event_locked("unload", model_id, reason)
        return {
            "model_id": model_id,
            "loaded": False,
            "reason": reason,
            "result": result,
        }

    def _lru_evictable_models_locked(
        self,
        *,
        exclude_model_id: Optional[str] = None,
    ) -> List[tuple[float, str]]:
        candidates: List[tuple[float, str]] = []
        for model_id, handle in self._loaded_items_locked():
            if model_id == exclude_model_id or handle.registration.pinned:
                continue
            score = handle.last_used_at or handle.loaded_at or 0.0
            candidates.append((score, model_id))
        candidates.sort()
        return candidates

    def _enforce_policies_locked(
        self,
        *,
        exclude_model_id: Optional[str] = None,
    ) -> None:
        now = time.time()
        for model_id, handle in list(self._models.items()):
            if not handle.loaded or handle.registration.pinned or model_id == exclude_model_id:
                continue
            ttl_seconds = handle.registration.ttl_seconds
            if ttl_seconds > 0 and handle.loaded_at is not None:
                if now - handle.loaded_at >= ttl_seconds:
                    self._unload_model_locked(model_id, reason="ttl_expired")
                    continue
            idle_unload_seconds = handle.registration.idle_unload_seconds
            if idle_unload_seconds > 0 and handle.last_used_at is not None:
                if now - handle.last_used_at >= idle_unload_seconds:
                    self._unload_model_locked(model_id, reason="idle_timeout")

    def _ensure_capacity_locked(self, target_model_id: str) -> None:
        while len(self._loaded_items_locked()) >= self._capacity_locked():
            candidates = self._lru_evictable_models_locked(exclude_model_id=target_model_id)
            if not candidates:
                raise BackendError(
                    "model pool is at capacity and all loaded models are pinned"
                )
            _, victim_model_id = candidates[0]
            self._unload_model_locked(victim_model_id, reason="lru_eviction")

    def load_model(self, model_id: str, *, reason: str = "manual_load") -> Dict[str, Any]:
        with self._lock:
            resolved_model_id = self._resolve_model_id(model_id)
            handle = self._require_handle_locked(resolved_model_id)
            self._enforce_policies_locked(exclude_model_id=resolved_model_id)
            if handle.loaded:
                handle.last_used_at = time.time()
                return {
                    "model_id": resolved_model_id,
                    "loaded": True,
                    "reason": "already_loaded",
                    "pool": self.model_pool_diagnostics(),
                }
            self._ensure_capacity_locked(resolved_model_id)
            result = handle.adapter.start_runtime()
            now = time.time()
            handle.loaded = True
            handle.loaded_at = now
            handle.last_used_at = now
            handle.last_load_reason = reason
            handle.last_unload_reason = ""
            self._record_event_locked("load", resolved_model_id, reason)
            return {
                "model_id": resolved_model_id,
                "loaded": True,
                "reason": reason,
                "result": result,
                "pool": self.model_pool_diagnostics(),
            }

    def unload_model(self, model_id: str, *, reason: str = "manual_unload") -> Dict[str, Any]:
        with self._lock:
            resolved_model_id = self._resolve_model_id(model_id)
            result = self._unload_model_locked(resolved_model_id, reason=reason)
            result["pool"] = self.model_pool_diagnostics()
            return result

    def set_model_pin(self, model_id: str, pinned: bool) -> Dict[str, Any]:
        with self._lock:
            resolved_model_id = self._resolve_model_id(model_id)
            handle = self._require_handle_locked(resolved_model_id)
            handle.registration.pinned = bool(pinned)
            self._record_event_locked(
                "pin",
                resolved_model_id,
                "pinned" if pinned else "unpinned",
            )
            return {
                "model_id": resolved_model_id,
                "pinned": handle.registration.pinned,
                "pool": self.model_pool_diagnostics(),
            }

    def _touch_model_locked(self, model_id: str) -> None:
        handle = self._require_handle_locked(model_id)
        now = time.time()
        handle.last_used_at = now
        if handle.loaded_at is None:
            handle.loaded_at = now
        handle.request_count += 1
        self._models.move_to_end(model_id)

    def model_pool_diagnostics(self) -> Dict[str, Any]:
        with self._lock:
            self._enforce_policies_locked()
            items: List[Dict[str, Any]] = []
            now = time.time()
            loaded_models: List[str] = []
            for model_id, handle in self._models.items():
                if handle.loaded:
                    loaded_models.append(model_id)
                ttl_remaining = None
                if handle.loaded and handle.loaded_at is not None and handle.registration.ttl_seconds > 0:
                    ttl_remaining = max(
                        0.0,
                        handle.registration.ttl_seconds - (now - handle.loaded_at),
                    )
                idle_remaining = None
                if (
                    handle.loaded
                    and handle.last_used_at is not None
                    and handle.registration.idle_unload_seconds > 0
                ):
                    idle_remaining = max(
                        0.0,
                        handle.registration.idle_unload_seconds - (now - handle.last_used_at),
                    )
                items.append(
                    {
                        "model_id": model_id,
                        "model_alias": handle.registration.model_alias,
                        "loaded": handle.loaded,
                        "pinned": handle.registration.pinned,
                        "base_url": handle.registration.base_url or handle.adapter.base_url,
                        "backend_kind": handle.registration.backend_kind,
                        "backend_model_name": handle.registration.backend_model_name,
                        "primary_service": handle.registration.primary_service,
                        "model_repo_id": handle.registration.model_repo_id,
                        "artifact_path": handle.registration.artifact_path,
                        "mmproj_path": handle.registration.mmproj_path,
                        "gguf_variant": handle.registration.gguf_variant,
                        "serving_preset": handle.registration.serving_preset,
                        "ctx_size": handle.registration.ctx_size
                        or getattr(getattr(handle.adapter, "config", None), "ctx_size", 0),
                        "parallel_slots": handle.registration.parallel_slots
                        or getattr(
                            getattr(handle.adapter, "config", None),
                            "parallel_slots",
                            0,
                        ),
                        "ttl_seconds": handle.registration.ttl_seconds,
                        "ttl_remaining_seconds": (
                            None if ttl_remaining is None else round(ttl_remaining, 3)
                        ),
                        "idle_unload_seconds": handle.registration.idle_unload_seconds,
                        "idle_remaining_seconds": (
                            None if idle_remaining is None else round(idle_remaining, 3)
                        ),
                        "loaded_at": None if handle.loaded_at is None else round(handle.loaded_at, 3),
                        "last_used_at": (
                            None if handle.last_used_at is None else round(handle.last_used_at, 3)
                        ),
                        "last_load_reason": handle.last_load_reason,
                        "last_unload_reason": handle.last_unload_reason,
                        "last_eviction_reason": handle.last_eviction_reason,
                        "request_count": handle.request_count,
                        "capabilities": {
                            "embeddings": handle.registration.supports_embeddings,
                            "rerank": handle.registration.supports_rerank,
                            "vision_chat": handle.registration.supports_vision,
                            "ocr": handle.registration.supports_ocr,
                        },
                    }
                )
            return {
                "enabled": True,
                "default_model_id": self._default_model_id,
                "max_loaded_models": self._capacity_locked(),
                "loaded_models": loaded_models,
                "models": items,
                "recent_events": [event.to_dict() for event in self._recent_events],
            }

    def _focus_model_id_locked(self) -> str:
        default_handle = self._models.get(self._default_model_id)
        if default_handle is not None and default_handle.loaded:
            return self._default_model_id
        for model_id, handle in self._models.items():
            if handle.loaded:
                return model_id
        if self._default_model_id in self._models:
            return self._default_model_id
        if self._models:
            return next(iter(self._models.keys()))
        return self._default_model_id

    def health(self) -> bool:
        with self._lock:
            self._enforce_policies_locked()
            loaded_handles = self._loaded_items_locked()
            if not loaded_handles:
                return False
            return any(handle.adapter.health() for _, handle in loaded_handles)

    def list_models(self) -> dict:
        with self._lock:
            data = []
            for model_id, handle in self._models.items():
                data.append(
                    {
                        "id": handle.registration.model_alias or model_id,
                        "root": model_id,
                        "loaded": handle.loaded,
                        "default": model_id == self._default_model_id,
                    }
                )
        return {"object": "list", "data": data}

    def proxy(self, method: str, path: str, **kwargs: Any):
        payload = kwargs.get("json")
        requested_model = payload.get("model") if isinstance(payload, dict) else None
        with self._lock:
            model_id = self._resolve_model_id(requested_model)
        load_result = self.load_model(model_id, reason="request")
        if not load_result.get("loaded"):
            raise BackendError(f"failed to load model: {model_id}")
        with self._lock:
            handle = self._require_handle_locked(model_id)
        if isinstance(payload, dict):
            payload = dict(payload)
            target_model_name = (
                handle.registration.backend_model_name
                or handle.registration.model_repo_id
                or model_id
            )
            payload["model"] = target_model_name
            kwargs["json"] = payload
        response = handle.adapter.proxy(method, path, **kwargs)
        with self._lock:
            self._touch_model_locked(model_id)
            self._enforce_policies_locked(exclude_model_id=model_id)
        return response

    def collect_metrics(self) -> RuntimeMetrics:
        with self._lock:
            self._enforce_policies_locked()
            focus_model_id = self._focus_model_id_locked()
            focus_handle = self._models.get(focus_model_id)
            loaded_handles = self._loaded_items_locked()
        if focus_handle is None:
            metrics = RuntimeMetrics(backend_url=self.base_url, healthy=False)
        else:
            metrics = focus_handle.adapter.collect_metrics()
        details = dict(metrics.details or {})
        details["model_pool"] = self.model_pool_diagnostics()
        details["focus_model_id"] = focus_model_id
        metrics.details = details
        if loaded_handles:
            metrics.healthy = any(handle.adapter.health() for _, handle in loaded_handles)
        return metrics

    def start_runtime(self) -> Dict[str, Any]:
        return self.load_model(self._default_model_id, reason="manual_start")

    def stop_runtime(self) -> Dict[str, Any]:
        with self._lock:
            stopped = []
            for model_id, handle in list(self._models.items()):
                if handle.loaded:
                    stopped.append(self._unload_model_locked(model_id, reason="manual_stop"))
            return {
                "mode": "llama_cpp_pool",
                "stopped_models": stopped,
                "pool": self.model_pool_diagnostics(),
            }

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        with self._lock:
            logs = {
                model_id: handle.adapter.runtime_logs(lines=lines)
                for model_id, handle in self._models.items()
                if handle.loaded
            }
        return {
            "mode": "llama_cpp_pool",
            "logs": logs,
            "pool": self.model_pool_diagnostics(),
        }

    def cache_report(self) -> Dict[str, Any]:
        with self._lock:
            self._enforce_policies_locked()
            focus_model_id = self._focus_model_id_locked()
            focus_handle = self._models.get(focus_model_id)
        payload: Dict[str, Any] = {
            "model_pool": self.model_pool_diagnostics(),
            "focus_model_id": focus_model_id,
        }
        if focus_handle is not None and focus_handle.loaded:
            payload["focus_model"] = focus_handle.adapter.cache_report()
        return payload

    def capabilities(self) -> BackendCapabilities:
        aggregate = BackendCapabilities()
        for handle in self._models.values():
            capabilities = handle.adapter.capabilities()
            registration = handle.registration
            aggregate = BackendCapabilities(
                chat_completions=aggregate.chat_completions or capabilities.chat_completions,
                completions=aggregate.completions or capabilities.completions,
                embeddings=aggregate.embeddings or (
                    capabilities.embeddings and registration.supports_embeddings
                ),
                rerank=aggregate.rerank or (
                    capabilities.rerank and registration.supports_rerank
                ),
                vision_chat=aggregate.vision_chat or (
                    capabilities.vision_chat and registration.supports_vision
                ),
                ocr=aggregate.ocr or (capabilities.ocr and registration.supports_ocr),
            )
        if not self._models:
            return BackendCapabilities()
        return BackendCapabilities(
            chat_completions=aggregate.chat_completions,
            completions=aggregate.completions,
            embeddings=aggregate.embeddings,
            rerank=aggregate.rerank,
            vision_chat=aggregate.vision_chat,
            ocr=aggregate.ocr,
        )
