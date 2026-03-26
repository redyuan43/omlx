#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark concurrent Qwen3.5-35B requests against local Ollama."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
import urllib.error
from typing import Any, Dict, List

import bench_qwen35_35b_ollama as single_bench


def _chat_request(
    *,
    base_url: str,
    model: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    messages = payload["messages"]
    options = payload["options"]
    result = single_bench._run_chat(
        base_url=base_url,
        model=model,
        messages=messages,
        num_ctx=options["num_ctx"],
        num_predict=options["num_predict"],
        keep_alive=payload["keep_alive"],
        timeout=payload["timeout"],
        think=payload["think"],
    )
    return result


def _build_payload(
    *,
    model: str,
    content: str,
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    timeout: int,
    think: bool,
) -> Dict[str, Any]:
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        "keep_alive": keep_alive,
        "timeout": timeout,
        "think": think,
    }


def _run_parallel_chat(
    *,
    base_url: str,
    requests_to_run: List[Dict[str, Any]],
) -> Dict[str, Any]:
    started = time.perf_counter()
    barrier = threading.Barrier(len(requests_to_run))
    results: Dict[str, Dict[str, Any]] = {}

    def _worker(case: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        barrier.wait()
        result = _chat_request(
            base_url=base_url,
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
    eval_count = sum(
        item.get("eval_count", 0) or 0
        for item in results.values()
    )
    return {
        "makespan_sec": round(makespan, 3),
        "aggregate_token_per_second": (
            round(eval_count / max(makespan, 1e-6), 3)
            if eval_count
            else None
        ),
        "requests": results,
    }


def _long_prefix_prompt(prefix_unit: str, repeat: int) -> str:
    return f"{' '.join([prefix_unit] * repeat)}\n\nReply with exactly OK."


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark concurrent Ollama Qwen3.5-35B requests")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:35b-q4km-local")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--target-context-tokens", type=int, default=32768)
    parser.add_argument("--long-prefix-repeat", type=int, default=0)
    parser.add_argument("--long-output-num-predict", type=int, default=192)
    parser.add_argument("--keep-alive", default="4h")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prefix-salt", default="")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    try:
        version = single_bench._http_json(f"{args.base_url}/api/version", timeout=30)
        tags = single_bench._http_json(f"{args.base_url}/api/tags", timeout=30)
    except urllib.error.URLError as exc:
        raise SystemExit(f"ollama unavailable: {exc}") from exc

    model_details = single_bench._show_model(args.base_url, args.model)

    prefix_base = "Concurrent Ollama Qwen3.5 35B cache benchmark"
    if args.prefix_salt:
        prefix_base = f"{prefix_base} {args.prefix_salt}"

    effective_context_tokens = min(args.target_context_tokens, args.num_ctx)
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
        long_prefix_estimated_tokens = single_bench._estimate_prompt_tokens(long_prefix_a)
    else:
        context_headroom = max(4096, min(16384, effective_context_tokens // 4))
        long_prefix_budget = max(1024, effective_context_tokens - context_headroom)
        long_prefix_a_raw, long_prefix_repeat, long_prefix_estimated_tokens = single_bench._build_long_prefix(
            f"{prefix_base} session-a",
            long_prefix_budget,
        )
        long_prefix_b_raw, _, _ = single_bench._build_long_prefix(
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
                num_ctx=args.num_ctx,
                num_predict=16,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                think=args.think,
            ),
        },
        {
            "label": "short_b",
            "payload": _build_payload(
                model=args.model,
                content="Reply with exactly PONG.",
                num_ctx=args.num_ctx,
                num_predict=16,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                think=args.think,
            ),
        },
    ]
    long_output_requests = [
        {
            "label": "long_output_a",
            "payload": _build_payload(
                model=args.model,
                content=long_output_prompt,
                num_ctx=args.num_ctx,
                num_predict=args.long_output_num_predict,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                think=args.think,
            ),
        },
        {
            "label": "long_output_b",
            "payload": _build_payload(
                model=args.model,
                content=long_output_prompt,
                num_ctx=args.num_ctx,
                num_predict=args.long_output_num_predict,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                think=args.think,
            ),
        },
    ]
    long_prefix_requests = [
        {
            "label": "long_prefix_a",
            "payload": _build_payload(
                model=args.model,
                content=long_prefix_a,
                num_ctx=args.num_ctx,
                num_predict=4,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                think=args.think,
            ),
        },
        {
            "label": "long_prefix_b",
            "payload": _build_payload(
                model=args.model,
                content=long_prefix_b,
                num_ctx=args.num_ctx,
                num_predict=4,
                keep_alive=args.keep_alive,
                timeout=args.timeout,
                think=args.think,
            ),
        },
    ]

    warmup_payload = _build_payload(
        model=args.model,
        content=long_prefix_a,
        num_ctx=args.num_ctx,
        num_predict=4,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    warmup = _chat_request(
        base_url=args.base_url,
        model=args.model,
        payload=warmup_payload,
    )
    repeat_plus_short = _run_parallel_chat(
        base_url=args.base_url,
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
                    num_ctx=args.num_ctx,
                    num_predict=16,
                    keep_alive=args.keep_alive,
                    timeout=args.timeout,
                    think=args.think,
                ),
            },
        ],
    )
    dual_short = _run_parallel_chat(
        base_url=args.base_url,
        requests_to_run=short_requests,
    )
    dual_long_output = _run_parallel_chat(
        base_url=args.base_url,
        requests_to_run=long_output_requests,
    )
    dual_long_prefix = _run_parallel_chat(
        base_url=args.base_url,
        requests_to_run=long_prefix_requests,
    )

    results = {
        "base_url": args.base_url,
        "version": version,
        "installed_models": [item.get("name") for item in tags.get("models", []) if item.get("name")],
        "model": args.model,
        "model_details": model_details,
        "think_enabled": args.think,
        "benchmark_context": {
            "requested_num_ctx": args.num_ctx,
            "target_context_tokens": args.target_context_tokens,
            "effective_context_tokens": effective_context_tokens,
            "long_prefix_repeat": long_prefix_repeat,
            "long_prefix_estimated_tokens": long_prefix_estimated_tokens,
            "keep_alive": args.keep_alive,
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
