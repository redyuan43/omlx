# SPDX-License-Identifier: Apache-2.0
"""Adaptive routing across multiple managed runtimes."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict, deque
from dataclasses import asdict, dataclass
from typing import Any, Deque, Dict, Optional
from urllib.parse import ParseResult, urlparse

import requests

from .backend import BackendAdapter, BackendCapabilities, BackendError, RuntimeMetrics


def derive_secondary_base_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = (parsed.port or 30000) + 1
    netloc = f"{host}:{port}"
    if parsed.username:
        credentials = parsed.username
        if parsed.password:
            credentials = f"{credentials}:{parsed.password}"
        netloc = f"{credentials}@{netloc}"
    updated = ParseResult(
        scheme=parsed.scheme or "http",
        netloc=netloc,
        path=parsed.path,
        params=parsed.params,
        query=parsed.query,
        fragment=parsed.fragment,
    )
    return updated.geturl()


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
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
        return json_safe_dumps(value)
    return str(value)


def _extract_prompt_text(payload: Dict[str, Any]) -> str:
    if isinstance(payload.get("messages"), list):
        parts = []
        for message in payload["messages"]:
            if not isinstance(message, dict):
                continue
            role = message.get("role", "user")
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


def _hash_key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


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


def json_safe_dumps(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False)


@dataclass
class RouteDecision:
    backend_label: str
    backend_url: str
    chunked_prefill_size: int
    estimated_prompt_tokens: int
    reason: str
    request_path: str
    sticky_key_hash: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = round(self.timestamp, 3)
        return payload


class AdaptiveBackendAdapter(BackendAdapter):
    """Routes requests between two backends using a sticky long-context policy."""

    def __init__(
        self,
        *,
        primary: BackendAdapter,
        secondary: BackendAdapter,
        primary_url: str,
        secondary_url: str,
        primary_chunked_prefill_size: int,
        secondary_chunked_prefill_size: int,
        short_prompt_threshold: int,
        max_sticky_sessions: int = 256,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.primary_url = primary_url
        self.secondary_url = secondary_url
        self.primary_chunked_prefill_size = primary_chunked_prefill_size
        self.secondary_chunked_prefill_size = secondary_chunked_prefill_size
        self.short_prompt_threshold = short_prompt_threshold
        self.max_sticky_sessions = max_sticky_sessions
        self._sticky_small_sessions: "OrderedDict[str, float]" = OrderedDict()
        self._route_counts: Dict[str, int] = {
            "primary": 0,
            "secondary": 0,
            "short_prompt": 0,
            "long_prompt": 0,
            "sticky_session": 0,
            "default": 0,
        }
        self._recent_decisions: Deque[RouteDecision] = deque(maxlen=20)
        self._last_decision: Optional[RouteDecision] = None

    def _remember_small_session(self, routing_key: str) -> None:
        self._sticky_small_sessions.pop(routing_key, None)
        self._sticky_small_sessions[routing_key] = time.time()
        while len(self._sticky_small_sessions) > self.max_sticky_sessions:
            self._sticky_small_sessions.popitem(last=False)

    def _select_backend(
        self,
        *,
        path: str,
        payload: Optional[Dict[str, Any]],
    ) -> tuple[BackendAdapter, RouteDecision]:
        if not payload or path not in {"v1/chat/completions", "v1/completions"}:
            decision = RouteDecision(
                backend_label="primary",
                backend_url=self.primary_url,
                chunked_prefill_size=self.primary_chunked_prefill_size,
                estimated_prompt_tokens=0,
                reason="default",
                request_path=path,
                sticky_key_hash="",
                timestamp=time.time(),
            )
            return self.primary, decision

        estimated_prompt_tokens = _estimate_prompt_tokens(payload)
        routing_key = _extract_routing_key(payload)
        if routing_key and routing_key in self._sticky_small_sessions:
            self._sticky_small_sessions.move_to_end(routing_key)
            decision = RouteDecision(
                backend_label="secondary",
                backend_url=self.secondary_url,
                chunked_prefill_size=self.secondary_chunked_prefill_size,
                estimated_prompt_tokens=estimated_prompt_tokens,
                reason="sticky_session",
                request_path=path,
                sticky_key_hash=_hash_key(routing_key),
                timestamp=time.time(),
            )
            return self.secondary, decision

        if estimated_prompt_tokens >= self.short_prompt_threshold:
            if routing_key:
                self._remember_small_session(routing_key)
            decision = RouteDecision(
                backend_label="secondary",
                backend_url=self.secondary_url,
                chunked_prefill_size=self.secondary_chunked_prefill_size,
                estimated_prompt_tokens=estimated_prompt_tokens,
                reason="long_prompt",
                request_path=path,
                sticky_key_hash=_hash_key(routing_key) if routing_key else "",
                timestamp=time.time(),
            )
            return self.secondary, decision

        decision = RouteDecision(
            backend_label="primary",
            backend_url=self.primary_url,
            chunked_prefill_size=self.primary_chunked_prefill_size,
            estimated_prompt_tokens=estimated_prompt_tokens,
            reason="short_prompt",
            request_path=path,
            sticky_key_hash=_hash_key(routing_key) if routing_key else "",
            timestamp=time.time(),
        )
        return self.primary, decision

    def _record_decision(self, decision: RouteDecision) -> None:
        self._last_decision = decision
        self._recent_decisions.append(decision)
        self._route_counts[decision.backend_label] += 1
        self._route_counts[decision.reason] += 1

    def _routing_details(self) -> Dict[str, Any]:
        return {
            "strategy": "adaptive",
            "short_prompt_threshold": self.short_prompt_threshold,
            "sticky_small_session_count": len(self._sticky_small_sessions),
            "profiles": {
                "primary": {
                    "backend_url": self.primary_url,
                    "chunked_prefill_size": self.primary_chunked_prefill_size,
                },
                "secondary": {
                    "backend_url": self.secondary_url,
                    "chunked_prefill_size": self.secondary_chunked_prefill_size,
                },
            },
            "route_counts": dict(self._route_counts),
            "last_decision": (
                self._last_decision.to_dict() if self._last_decision is not None else None
            ),
            "recent_decisions": [decision.to_dict() for decision in self._recent_decisions],
        }

    def health(self) -> bool:
        return self.primary.health() and self.secondary.health()

    def list_models(self) -> dict:
        return self.primary.list_models()

    def proxy(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        payload = kwargs.get("json")
        backend, decision = self._select_backend(path=path, payload=payload)
        self._record_decision(decision)
        try:
            return backend.proxy(method, path, **kwargs)
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(str(exc)) from exc

    def collect_metrics(self) -> RuntimeMetrics:
        def pick(primary_value: Any, secondary_value: Any) -> Any:
            return primary_value if primary_value is not None else secondary_value

        primary_metrics = self.primary.collect_metrics()
        secondary_metrics = self.secondary.collect_metrics()
        details = {
            "routing": self._routing_details(),
            "backends": {
                "primary": primary_metrics.to_dict(),
                "secondary": secondary_metrics.to_dict(),
            },
        }
        if primary_metrics.details:
            details["primary_details"] = primary_metrics.details
        if secondary_metrics.details:
            details["secondary_details"] = secondary_metrics.details
        return RuntimeMetrics(
            backend_url=self.primary_url,
            healthy=primary_metrics.healthy and secondary_metrics.healthy,
            gpu_name=pick(primary_metrics.gpu_name, secondary_metrics.gpu_name),
            gpu_memory_used_mb=pick(
                primary_metrics.gpu_memory_used_mb,
                secondary_metrics.gpu_memory_used_mb,
            ),
            gpu_memory_total_mb=pick(
                primary_metrics.gpu_memory_total_mb,
                secondary_metrics.gpu_memory_total_mb,
            ),
            gpu_util_percent=pick(
                primary_metrics.gpu_util_percent,
                secondary_metrics.gpu_util_percent,
            ),
            gpu_temperature_c=pick(
                primary_metrics.gpu_temperature_c,
                secondary_metrics.gpu_temperature_c,
            ),
            details=details,
        )

    def start_runtime(self) -> Dict[str, Any]:
        primary_started = self.primary.start_runtime()
        try:
            secondary_started = self.secondary.start_runtime()
        except Exception:
            try:
                self.primary.stop_runtime()
            except Exception:
                pass
            raise
        return {
            "mode": "adaptive",
            "started": bool(primary_started.get("started") or secondary_started.get("started")),
            "backends": {
                "primary": primary_started,
                "secondary": secondary_started,
            },
        }

    def stop_runtime(self) -> Dict[str, Any]:
        return {
            "mode": "adaptive",
            "backends": {
                "primary": self.primary.stop_runtime(),
                "secondary": self.secondary.stop_runtime(),
            },
        }

    def runtime_logs(self, lines: int = 40) -> Dict[str, Any]:
        return {
            "mode": "adaptive",
            "backends": {
                "primary": self.primary.runtime_logs(lines=lines),
                "secondary": self.secondary.runtime_logs(lines=lines),
            },
        }

    def hicache_storage_status(self) -> Dict[str, Any]:
        return {
            "mode": "adaptive",
            "backends": {
                "primary": self.primary.hicache_storage_status(),
                "secondary": self.secondary.hicache_storage_status(),
            },
        }

    def attach_hicache_storage_backend(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return {
            "mode": "adaptive",
            "backends": {
                "primary": self.primary.attach_hicache_storage_backend(overrides=overrides),
                "secondary": self.secondary.attach_hicache_storage_backend(overrides=overrides),
            },
        }

    def detach_hicache_storage_backend(self) -> Dict[str, Any]:
        return {
            "mode": "adaptive",
            "backends": {
                "primary": self.primary.detach_hicache_storage_backend(),
                "secondary": self.secondary.detach_hicache_storage_backend(),
            },
        }

    def cache_report(self) -> Dict[str, Any]:
        return {
            "mode": "adaptive",
            "backends": {
                "primary": self.primary.cache_report(),
                "secondary": self.secondary.cache_report(),
            },
        }

    def capabilities(self) -> BackendCapabilities:
        primary = self.primary.capabilities()
        secondary = self.secondary.capabilities()
        return BackendCapabilities(
            chat_completions=primary.chat_completions and secondary.chat_completions,
            completions=primary.completions and secondary.completions,
            embeddings=primary.embeddings,
            rerank=primary.rerank,
            vision_chat=primary.vision_chat and secondary.vision_chat,
            ocr=primary.ocr and secondary.ocr,
        )
