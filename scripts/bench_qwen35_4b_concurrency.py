#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark concurrent Qwen3.5-4B requests against the local control plane."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
from typing import Any, Dict, List

import bench_qwen35_4b as single_bench


def _effective_slot_context_tokens(
    runtime_summary: Dict[str, Any],
    requested_context_tokens: int,
) -> int:
    loaded_context_length = runtime_summary.get("loaded_context_length")
    effective_context_tokens = (
        min(requested_context_tokens, loaded_context_length)
        if isinstance(loaded_context_length, int) and loaded_context_length > 0
        else requested_context_tokens
    )
    backend_format = runtime_summary.get("backend_format")
    ctx_size = runtime_summary.get("ctx_size")
    parallel_slots = runtime_summary.get("parallel_slots")
    if (
        backend_format == "llama_cpp_gguf"
        and isinstance(ctx_size, int)
        and ctx_size > 0
        and isinstance(parallel_slots, int)
        and parallel_slots > 1
    ):
        slot_context_tokens = max(1024, ctx_size // parallel_slots)
        effective_context_tokens = min(effective_context_tokens, slot_context_tokens)
    return effective_context_tokens


def _chat_request(
    *,
    control_plane_url: str,
    model: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    started = time.perf_counter()
    response = single_bench._http_json(  # type: ignore[attr-defined]
        f"{control_plane_url}/v1/chat/completions",
        method="POST",
        payload=payload,
    )
    wall_time = time.perf_counter() - started
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0) or 0
    return {
        "wall_time_sec": round(wall_time, 3),
        "token_per_second": (
            round(completion_tokens / max(wall_time, 1e-6), 3)
            if completion_tokens
            else None
        ),
        "usage": usage,
        "cached_prompt_tokens": (
            usage.get("prompt_tokens_details", {}) or {}
        ).get("cached_tokens"),
        "assistant": response["choices"][0]["message"],
    }


def _build_payload(
    *,
    model: str,
    content: str | None = None,
    messages: List[Dict[str, Any]] | None = None,
    max_tokens: int,
    conversation_id: str,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "metadata": {"conversation_id": conversation_id},
        "chat_template_kwargs": {"enable_thinking": False},
        "enableThinking": False,
        "reasoning": False,
    }
    if messages is not None:
        payload["messages"] = messages
    else:
        payload["messages"] = [{"role": "user", "content": content or ""}]
    return payload


def _run_parallel_chat(
    *,
    control_plane_url: str,
    requests_to_run: List[Dict[str, Any]],
) -> Dict[str, Any]:
    started = time.perf_counter()
    barrier = threading.Barrier(len(requests_to_run))
    results: Dict[str, Dict[str, Any]] = {}

    def _worker(case: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        barrier.wait()
        result = _chat_request(
            control_plane_url=control_plane_url,
            model=case["payload"]["model"],
            payload=case["payload"],
        )
        return case["label"], result

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests_to_run)) as executor:
        futures = [executor.submit(_worker, case) for case in requests_to_run]
        for future in concurrent.futures.as_completed(futures):
            label, result = future.result()
            results[label] = result

    makespan = time.perf_counter() - started
    completion_tokens = sum(
        (item.get("usage", {}) or {}).get("completion_tokens", 0) or 0
        for item in results.values()
    )
    return {
        "makespan_sec": round(makespan, 3),
        "aggregate_token_per_second": (
            round(completion_tokens / max(makespan, 1e-6), 3)
            if completion_tokens
            else None
        ),
        "requests": results,
    }


def _long_prefix_prompt(prefix_unit: str, repeat: int) -> str:
    return f"{' '.join([prefix_unit] * repeat)}\n\nReply with exactly OK."


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark concurrent Qwen3.5-4B requests")
    parser.add_argument("--control-plane-url", default="http://127.0.0.1:8020")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:31200")
    parser.add_argument("--model", default="qwen35-4b-gguf")
    parser.add_argument("--long-prefix-repeat", type=int, default=0)
    parser.add_argument("--long-output-max-tokens", type=int, default=192)
    parser.add_argument("--target-context-tokens", type=int, default=65536)
    parser.add_argument("--prefix-salt", default="")
    args = parser.parse_args()

    try:
        health = single_bench._http_json(f"{args.control_plane_url}/health")  # type: ignore[attr-defined]
    except urllib.error.URLError as exc:
        raise SystemExit(f"control plane unavailable: {exc}") from exc

    runtime_before = single_bench._http_json(  # type: ignore[attr-defined]
        f"{args.control_plane_url}/admin/api/runtime"
    )
    runtime_summary = single_bench._runtime_summary(runtime_before)  # type: ignore[attr-defined]
    effective_context_tokens = _effective_slot_context_tokens(
        runtime_summary,
        args.target_context_tokens,
    )
    loaded_context_length = runtime_summary.get("loaded_context_length")

    prefix_base = "Concurrent llama.cpp Qwen3.5 cache benchmark"
    if args.prefix_salt:
        prefix_base = f"{prefix_base} {args.prefix_salt}"
    if args.long_prefix_repeat > 0:
        long_prefix_a = _long_prefix_prompt(
            f"{prefix_base} session-a",
            args.long_prefix_repeat,
        )
        long_prefix_b = _long_prefix_prompt(
            f"{prefix_base} session-b",
            args.long_prefix_repeat,
        )
        long_prefix_repeat = args.long_prefix_repeat
        long_prefix_estimated_tokens = single_bench._estimate_prompt_tokens(  # type: ignore[attr-defined]
            long_prefix_a
        )
    else:
        context_headroom = max(
            4096,
            min(16384, effective_context_tokens // 4),
        )
        long_prefix_budget = max(
            1024,
            effective_context_tokens - context_headroom,
        )
        long_prefix_a_raw, long_prefix_repeat, long_prefix_estimated_tokens = (
            single_bench._build_long_prefix(  # type: ignore[attr-defined]
                f"{prefix_base} session-a",
                long_prefix_budget,
            )
        )
        long_prefix_b_raw, _, _ = single_bench._build_long_prefix(  # type: ignore[attr-defined]
            f"{prefix_base} session-b",
            long_prefix_budget,
        )
        long_prefix_a = f"{long_prefix_a_raw}\n\nReply with exactly OK."
        long_prefix_b = f"{long_prefix_b_raw}\n\nReply with exactly OK."
    long_output_prompt = (
        "Write a numbered list from 1 to 64. "
        "Each line must be short and in the format '<n>. cache benchmark'."
    )

    short_requests = [
        {
            "label": "short_a",
            "payload": _build_payload(
                model=args.model,
                content="Reply with exactly PONG.",
                max_tokens=16,
                conversation_id="dual-short-a",
            ),
        },
        {
            "label": "short_b",
            "payload": _build_payload(
                model=args.model,
                content="Reply with exactly PONG.",
                max_tokens=16,
                conversation_id="dual-short-b",
            ),
        },
    ]
    long_output_requests = [
        {
            "label": "long_output_a",
            "payload": _build_payload(
                model=args.model,
                content=long_output_prompt,
                max_tokens=args.long_output_max_tokens,
                conversation_id="dual-output-a",
            ),
        },
        {
            "label": "long_output_b",
            "payload": _build_payload(
                model=args.model,
                content=long_output_prompt,
                max_tokens=args.long_output_max_tokens,
                conversation_id="dual-output-b",
            ),
        },
    ]
    long_prefix_requests = [
        {
            "label": "long_prefix_a",
            "payload": _build_payload(
                model=args.model,
                content=long_prefix_a,
                max_tokens=4,
                conversation_id="dual-prefix-a",
            ),
        },
        {
            "label": "long_prefix_b",
            "payload": _build_payload(
                model=args.model,
                content=long_prefix_b,
                max_tokens=4,
                conversation_id="dual-prefix-b",
            ),
        },
    ]

    warmup_payload = _build_payload(
        model=args.model,
        content=long_prefix_a,
        max_tokens=4,
        conversation_id="repeat-prefix-a",
    )
    warmup = _chat_request(
        control_plane_url=args.control_plane_url,
        model=args.model,
        payload=warmup_payload,
    )
    repeat_plus_short = _run_parallel_chat(
        control_plane_url=args.control_plane_url,
        requests_to_run=[
            {
                "label": "repeat_long_prefix",
                "payload": warmup_payload,
            },
            {
                "label": "concurrent_short",
                "payload": _build_payload(
                    model=args.model,
                    content="Reply with exactly PONG.",
                    max_tokens=16,
                    conversation_id="repeat-prefix-short",
                ),
            },
        ],
    )

    dual_short = _run_parallel_chat(
        control_plane_url=args.control_plane_url,
        requests_to_run=short_requests,
    )
    dual_long_output = _run_parallel_chat(
        control_plane_url=args.control_plane_url,
        requests_to_run=long_output_requests,
    )
    dual_long_prefix = _run_parallel_chat(
        control_plane_url=args.control_plane_url,
        requests_to_run=long_prefix_requests,
    )
    runtime_after = single_bench._http_json(  # type: ignore[attr-defined]
        f"{args.control_plane_url}/admin/api/runtime"
    )
    results = {
        "urls": {
            "control_plane_url": args.control_plane_url,
            "runtime_url": args.runtime_url,
        },
        "health": health,
        "runtime_summary": runtime_summary,
        "benchmark_context": {
            "requested_context_tokens": args.target_context_tokens,
            "effective_context_tokens": effective_context_tokens,
            "loaded_context_length": loaded_context_length,
            "long_prefix_repeat": long_prefix_repeat,
            "long_prefix_estimated_tokens": long_prefix_estimated_tokens,
        },
        "runtime_after": {
            "slot_router": (
                runtime_after.get("backend", {})
                .get("details", {})
                .get("slot_router")
            ),
            "cache_report": runtime_after.get("cache_report"),
        },
        "dual_short_independent": dual_short,
        "dual_long_output_independent": dual_long_output,
        "dual_long_prefix_independent": dual_long_prefix,
        "repeat_long_prefix_plus_short": {
            "warmup": warmup,
            "parallel": repeat_plus_short,
            "repeat_speedup_x": round(
                warmup["wall_time_sec"] / max(
                    repeat_plus_short["requests"]["repeat_long_prefix"]["wall_time_sec"],
                    1e-6,
                ),
                2,
            ),
        },
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
