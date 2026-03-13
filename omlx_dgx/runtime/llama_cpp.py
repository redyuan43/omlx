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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional
from urllib.parse import urlparse

from omlx_dgx.config import BackendConfig

from .backend import BackendError, HttpOpenAIBackendAdapter, RuntimeMetrics


def _stringify_command(args: List[str]) -> str:
    return shlex.join(args)


_RECYCLE_OWNED_IDLE_SLOT_MIN_AGE_SECONDS = 1.0


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
    artifact_summary: Dict[str, Any]
    gguf_variant: str
    ctx_size: int
    parallel_slots: int
    n_gpu_layers: int
    flash_attn: bool
    batch_size: int
    ubatch_size: int
    cache_ram_mib: int
    cache_reuse: int
    checkpoint_every_n_tokens: int
    ctx_checkpoints: int
    slot_prompt_similarity: float
    enable_runtime_metrics: bool
    enable_session_stickiness: bool
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
        parsed = urlparse(config.base_url)
        self.launch_host = parsed.hostname or "127.0.0.1"
        self.launch_port = parsed.port or 30000
        self._lock = threading.RLock()
        self._session_bindings: "OrderedDict[str, SessionBinding]" = OrderedDict()
        self._slot_owners: Dict[int, str] = {}
        self._recent_slot_decisions: Deque[SlotRouteDecision] = deque(maxlen=32)
        self._last_slot_decision: Optional[SlotRouteDecision] = None
        self._slot_route_counts: Dict[str, int] = {
            "pass_through": 0,
            "explicit_slot": 0,
            "sticky_existing": 0,
            "sticky_new_long_prompt": 0,
            "sticky_no_idle_slot": 0,
            "short_prompt": 0,
            "short_prompt_unowned_slot": 0,
            "short_prompt_recycled_slot": 0,
            "no_routing_key": 0,
            "unkeyed_idle_slot": 0,
            "unkeyed_recycled_slot": 0,
        }

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
        args.extend(["--flash-attn", "on" if self.config.flash_attn else "off"])
        if self.config.cache_reuse > 0:
            args.extend(["--cache-reuse", str(self.config.cache_reuse)])
        if self.config.enable_runtime_metrics:
            args.append("--metrics")
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
            cache_reuse=self.config.cache_reuse,
            checkpoint_every_n_tokens=self.config.checkpoint_every_n_tokens,
            ctx_checkpoints=self.config.ctx_checkpoints,
            slot_prompt_similarity=self.config.slot_prompt_similarity,
            enable_runtime_metrics=self.config.enable_runtime_metrics,
            enable_session_stickiness=self.config.enable_session_stickiness,
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

    def _slot_router_summary(self, slots_payload: Dict[str, Any]) -> Dict[str, Any]:
        with self._lock:
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
        prepared = dict(payload)
        if path not in {"v1/chat/completions", "v1/completions"}:
            return prepared
        if prepared.get("cache_prompt") is None:
            prepared["cache_prompt"] = True
        if self.config.parallel_slots <= 1 or not self.config.enable_session_stickiness:
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
            binding = self._session_bindings.get(routing_key)
            if binding and binding.slot_id >= self.config.parallel_slots:
                self._session_bindings.pop(routing_key, None)
                self._slot_owners.pop(binding.slot_id, None)
                binding = None
            if binding:
                binding.last_used_at = time.time()
                binding.estimated_prompt_tokens = estimated_prompt_tokens
                self._session_bindings.move_to_end(routing_key)
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
        details: Dict[str, Any] = {
            "diagnostics": self.diagnostics().to_dict(),
        }
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
        payload = kwargs.get("json")
        if isinstance(payload, dict):
            kwargs["json"] = self._prepare_proxy_payload(path, payload)
        return super().proxy(method, path, **kwargs)

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
        return {**result, "mode": "llama_cpp"}

    def stop_runtime(self) -> Dict[str, Any]:
        return {**self.process_manager.stop(), "mode": "llama_cpp"}

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        return {**self.process_manager.logs(lines=lines), "mode": "llama_cpp"}

    def cache_report(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        props = self._request_optional_json("props")
        slots = self._request_optional_json("slots")
        if props:
            payload["props"] = props
        if slots:
            payload["slots"] = slots
            payload["slot_router"] = self._slot_router_summary(slots)
        return payload
