# SPDX-License-Identifier: Apache-2.0
"""Scheduler-policy primitives for DGX runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class RequestShape:
    """Features exposed to the policy layer for one queued request."""

    request_id: str
    prompt_tokens: int
    prefix_hit_tokens: int = 0
    retention_priority: int = 50
    has_external_restore: bool = False
    interactive: bool = True


@dataclass(frozen=True)
class AdmissionDecision:
    accepted: bool
    reason: str
    effective_tokens: int


@dataclass(frozen=True)
class EvictionCandidate:
    block_hash: str
    score: float
    reason: str


class OmlxSchedulerPolicy:
    """A compact policy layer mirroring the oMLX scheduling heuristics."""

    def __init__(
        self,
        *,
        max_batch_tokens: int = 32768,
        max_running_requests: int = 16,
        interactive_bonus: float = 20.0,
        restore_bonus: float = 15.0,
    ) -> None:
        self.max_batch_tokens = max_batch_tokens
        self.max_running_requests = max_running_requests
        self.interactive_bonus = interactive_bonus
        self.restore_bonus = restore_bonus

    def effective_cost(self, request: RequestShape) -> int:
        return max(1, request.prompt_tokens - request.prefix_hit_tokens)

    def admit(self, request: RequestShape, *, running_requests: int, running_tokens: int) -> AdmissionDecision:
        effective_tokens = self.effective_cost(request)
        if running_requests >= self.max_running_requests:
            return AdmissionDecision(False, "max_running_requests", effective_tokens)
        if running_tokens + effective_tokens > self.max_batch_tokens:
            return AdmissionDecision(False, "max_batch_tokens", effective_tokens)
        return AdmissionDecision(True, "accepted", effective_tokens)

    def sort_waiting(self, requests: Iterable[RequestShape]) -> List[RequestShape]:
        def key(request: RequestShape) -> tuple[float, int, int]:
            score = request.retention_priority
            if request.interactive:
                score += self.interactive_bonus
            if request.has_external_restore:
                score += self.restore_bonus
            score += request.prefix_hit_tokens / max(request.prompt_tokens, 1)
            return (-score, self.effective_cost(request), request.prompt_tokens)

        return sorted(requests, key=key)

    def rank_evictions(self, blocks: Iterable[dict]) -> List[EvictionCandidate]:
        candidates: List[EvictionCandidate] = []
        for block in blocks:
            age_penalty = float(block.get("last_access", 0.0))
            ref_penalty = float(block.get("ref_count", 0)) * 10.0
            tier_bonus = 25.0 if block.get("tier") == "gpu" else 0.0
            score = age_penalty + ref_penalty - tier_bonus
            candidates.append(
                EvictionCandidate(
                    block_hash=block.get("block_hash") or "",
                    score=score,
                    reason="higher score means safer to evict first",
                )
            )
        return sorted(candidates, key=lambda item: item.score)
