#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the local Ollama Qwen3.5-35B GGUF stack."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
    timeout: int = 1200,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(4):
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code not in {502, 503, 504} or attempt == 3:
                raise
            last_error = exc
        except urllib.error.URLError as exc:
            if attempt == 3:
                raise
            last_error = exc
        time.sleep(1.0 + attempt)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"request failed without response: {url}")


def _read_meminfo() -> Dict[str, int]:
    keys = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, _, raw_value = line.partition(":")
        if name not in keys:
            continue
        values[name] = int(raw_value.strip().split()[0])
    return values


def _estimate_prompt_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4, len(text.split()))


def _build_long_prefix(prefix_unit: str, target_tokens: int) -> tuple[str, int, int]:
    unit = prefix_unit.strip()
    repeat = 1
    prefix = unit
    estimated_tokens = _estimate_prompt_tokens(prefix)
    while estimated_tokens < target_tokens:
        repeat += 1
        prefix = " ".join([unit] * repeat)
        estimated_tokens = _estimate_prompt_tokens(prefix)
    return prefix, repeat, estimated_tokens


def _ns_to_s(value: Any) -> str:
    if not isinstance(value, int) or value <= 0:
        return "n/a"
    return f"{value / 1e9:.3f}s"


def _tps(tokens: Any, duration_ns: Any) -> str:
    if not isinstance(tokens, int) or not isinstance(duration_ns, int) or duration_ns <= 0:
        return "n/a"
    return f"{tokens / (duration_ns / 1e9):.2f} tok/s"


def _response_stats(response: Dict[str, Any], wall_time: float) -> Dict[str, Any]:
    prompt_eval_count = response.get("prompt_eval_count")
    prompt_eval_duration = response.get("prompt_eval_duration")
    eval_count = response.get("eval_count")
    eval_duration = response.get("eval_duration")
    total_duration = response.get("total_duration")
    load_duration = response.get("load_duration")
    message = response.get("message", {})
    content = message.get("content", "")
    thinking = message.get("thinking", "")
    return {
        "wall_time_sec": round(wall_time, 3),
        "total_duration_sec": _ns_to_s(total_duration),
        "load_duration_sec": _ns_to_s(load_duration),
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_sec": _ns_to_s(prompt_eval_duration),
        "eval_count": eval_count,
        "eval_duration_sec": _ns_to_s(eval_duration),
        "prompt_speed": _tps(prompt_eval_count, prompt_eval_duration),
        "decode_speed": _tps(eval_count, eval_duration),
        "token_per_second": (
            round(eval_count / max(wall_time, 1e-6), 3)
            if isinstance(eval_count, int) and eval_count > 0
            else None
        ),
        "answer_preview": (content.strip()[:160] + "...") if len(content.strip()) > 160 else content.strip(),
        "thinking_preview": (thinking.strip()[:160] + "...") if len(thinking.strip()) > 160 else thinking.strip(),
        "thinking_chars": len(thinking),
        "done": response.get("done"),
        "done_reason": response.get("done_reason"),
    }


def _show_model(base_url: str, model: str) -> Dict[str, Any]:
    try:
        return _http_json(
            f"{base_url}/api/show",
            method="POST",
            payload={"model": model},
            timeout=60,
        )
    except Exception as exc:
        return {"error": str(exc)}


def _run_chat(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, Any]],
    num_ctx: int,
    num_predict: int,
    keep_alive: str,
    timeout: int,
    think: bool,
) -> Dict[str, Any]:
    before = _read_meminfo()
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "think": think,
        "options": {
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "temperature": 0,
        },
        "messages": messages,
    }
    started = time.perf_counter()
    response = _http_json(
        f"{base_url}/api/chat",
        method="POST",
        payload=payload,
        timeout=timeout,
    )
    wall_time = time.perf_counter() - started
    after = _read_meminfo()
    stats = _response_stats(response, wall_time)
    stats.update(
        {
            "system_mem_before_kib": before,
            "system_mem_after_kib": after,
            "response": response,
            "think_enabled": think,
        }
    )
    return stats


def _assistant_message(result: Dict[str, Any]) -> Dict[str, Any]:
    response = result.get("response", {})
    message = response.get("message", {})
    return {
        "role": "assistant",
        "content": message.get("content", ""),
    }


def _restart_ollama_service(timeout_sec: int = 180) -> Dict[str, Any]:
    started = time.perf_counter()
    subprocess.run(["sudo", "systemctl", "restart", "ollama"], check=True)
    restart_wall = time.perf_counter() - started

    ready_started = time.perf_counter()
    last_error: Exception | None = None
    for _ in range(timeout_sec):
        try:
            _http_json("http://127.0.0.1:11434/api/version", timeout=10)
            ready_time = time.perf_counter() - ready_started
            return {
                "restart_wall_time_sec": round(restart_wall, 3),
                "ready_after_restart_sec": round(ready_time, 3),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(1)
    raise RuntimeError(f"ollama did not come back after restart: {last_error}")


def _run_single_session_followup(
    *,
    base_url: str,
    model: str,
    long_prompt: str,
    num_ctx: int,
    keep_alive: str,
    timeout: int,
    think: bool,
) -> Dict[str, Any]:
    turn1_messages = [
        {
            "role": "user",
            "content": f"{long_prompt}\n\nReply with exactly one digit: 1.",
        }
    ]
    turn1 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn1_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    turn2_messages = turn1_messages + [
        _assistant_message(turn1),
        {"role": "user", "content": "Reply with exactly one digit: 2."},
    ]
    turn2 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn2_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    turn3_messages = turn2_messages + [
        _assistant_message(turn2),
        {"role": "user", "content": "Reply with exactly one digit: 3."},
    ]
    turn3 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn3_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    followup_avg = None
    if turn2.get("wall_time_sec") is not None and turn3.get("wall_time_sec") is not None:
        followup_avg = round((turn2["wall_time_sec"] + turn3["wall_time_sec"]) / 2, 3)
    return {
        "turn1_long": turn1,
        "turn2_short_followup": turn2,
        "turn3_short_followup": turn3,
        "summary": {
            "turn1_long_sec": turn1.get("wall_time_sec"),
            "turn2_short_followup_sec": turn2.get("wall_time_sec"),
            "turn3_short_followup_sec": turn3.get("wall_time_sec"),
            "turn2_prompt_tokens": turn2.get("prompt_eval_count"),
            "turn3_prompt_tokens": turn3.get("prompt_eval_count"),
            "followup_avg_sec": followup_avg,
        },
    }


def _run_single_session_recovery(
    *,
    base_url: str,
    model: str,
    long_prompt: str,
    num_ctx: int,
    keep_alive: str,
    timeout: int,
    think: bool,
) -> Dict[str, Any]:
    turn1_messages = [
        {
            "role": "user",
            "content": f"{long_prompt}\n\nReply with exactly one digit: 1.",
        }
    ]
    turn1 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn1_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    turn2_messages = turn1_messages + [
        _assistant_message(turn1),
        {"role": "user", "content": "Reply with exactly one digit: 2."},
    ]
    turn2 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn2_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    restart = _restart_ollama_service()
    turn3_messages = turn2_messages + [
        _assistant_message(turn2),
        {"role": "user", "content": "After restart, reply with exactly one digit: 3."},
    ]
    turn3 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn3_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    turn4_messages = turn3_messages + [
        _assistant_message(turn3),
        {"role": "user", "content": "Reply with exactly one digit: 4."},
    ]
    turn4 = _run_chat(
        base_url=base_url,
        model=model,
        messages=turn4_messages,
        num_ctx=num_ctx,
        num_predict=1,
        keep_alive=keep_alive,
        timeout=timeout,
        think=think,
    )
    followup_avg = None
    if turn3.get("wall_time_sec") is not None and turn4.get("wall_time_sec") is not None:
        followup_avg = round((turn3["wall_time_sec"] + turn4["wall_time_sec"]) / 2, 3)
    return {
        "restart": restart,
        "turn1_long": turn1,
        "turn2_warm_followup": turn2,
        "turn3_post_restart_followup": turn3,
        "turn4_post_restore_followup": turn4,
        "summary": {
            "turn1_cold_long_sec": turn1.get("wall_time_sec"),
            "turn2_warm_followup_sec": turn2.get("wall_time_sec"),
            "turn3_post_restart_followup_sec": turn3.get("wall_time_sec"),
            "turn4_post_restore_followup_sec": turn4.get("wall_time_sec"),
            "followup_avg_sec": followup_avg,
            "restart_wall_time_sec": restart.get("restart_wall_time_sec"),
            "ready_after_restart_sec": restart.get("ready_after_restart_sec"),
        },
    }


def _print_section(title: str, payload: Dict[str, Any]) -> None:
    print(f"== {title} ==")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the local Ollama Qwen3.5-35B GGUF stack")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="qwen3.5:35b-q4km-local")
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--target-context-tokens", type=int, default=32768)
    parser.add_argument("--long-prefix-repeat", type=int, default=0)
    parser.add_argument("--short-num-predict", type=int, default=16)
    parser.add_argument("--long-output-num-predict", type=int, default=192)
    parser.add_argument("--keep-alive", default="4h")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--prefix-salt", default="")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    version = _http_json(f"{args.base_url}/api/version", timeout=30)
    tags = _http_json(f"{args.base_url}/api/tags", timeout=30)
    model_details = _show_model(args.base_url, args.model)

    prefix_unit = "Ollama Qwen3.5 35B cache benchmark"
    if args.prefix_salt:
        prefix_unit = f"{prefix_unit} {args.prefix_salt}"
    effective_context_tokens = min(args.target_context_tokens, args.num_ctx)
    if args.long_prefix_repeat > 0:
        long_prefix = " ".join([prefix_unit] * args.long_prefix_repeat)
        long_prefix_repeat = args.long_prefix_repeat
        long_prefix_estimated_tokens = _estimate_prompt_tokens(long_prefix)
    else:
        context_headroom = max(4096, min(16384, effective_context_tokens // 4))
        long_prefix_budget = max(1024, effective_context_tokens - context_headroom)
        long_prefix, long_prefix_repeat, long_prefix_estimated_tokens = _build_long_prefix(
            prefix_unit,
            long_prefix_budget,
        )
    long_prompt = f"{long_prefix}\n\nReply with exactly OK."
    long_output_prompt = (
        "Write a numbered list from 1 to 64. "
        "Each line must be short and in the format '<n>. cache benchmark'."
    )
    multi_turn_messages = [
        {
            "role": "system",
            "content": "You are a concise assistant. Keep each answer under 12 words.",
        },
        {
            "role": "user",
            "content": "Summarize why prefix reuse matters for long-context chat.",
        },
        {
            "role": "assistant",
            "content": "It avoids recomputing the same long prompt on every turn.",
        },
        {
            "role": "user",
            "content": "Now answer in one sentence: what is the main tradeoff of smaller prefill chunks?",
        },
    ]

    short_chat = _run_chat(
        base_url=args.base_url,
        model=args.model,
        messages=[{"role": "user", "content": "Reply with exactly PONG."}],
        num_ctx=args.num_ctx,
        num_predict=args.short_num_predict,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    long_output_chat = _run_chat(
        base_url=args.base_url,
        model=args.model,
        messages=[{"role": "user", "content": long_output_prompt}],
        num_ctx=args.num_ctx,
        num_predict=args.long_output_num_predict,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    long_prefix_run_1 = _run_chat(
        base_url=args.base_url,
        model=args.model,
        messages=[{"role": "user", "content": long_prompt}],
        num_ctx=args.num_ctx,
        num_predict=4,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    long_prefix_run_2 = _run_chat(
        base_url=args.base_url,
        model=args.model,
        messages=[{"role": "user", "content": long_prompt}],
        num_ctx=args.num_ctx,
        num_predict=4,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    multi_turn_chat = _run_chat(
        base_url=args.base_url,
        model=args.model,
        messages=multi_turn_messages,
        num_ctx=args.num_ctx,
        num_predict=64,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    single_session_followup = _run_single_session_followup(
        base_url=args.base_url,
        model=args.model,
        long_prompt=long_prefix,
        num_ctx=args.num_ctx,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )
    single_session_recovery = _run_single_session_recovery(
        base_url=args.base_url,
        model=args.model,
        long_prompt=long_prefix,
        num_ctx=args.num_ctx,
        keep_alive=args.keep_alive,
        timeout=args.timeout,
        think=args.think,
    )

    results = {
        "base_url": args.base_url,
        "model": args.model,
        "think_enabled": args.think,
        "version": version,
        "tags": tags,
        "model_details": model_details,
        "benchmark_context": {
            "requested_num_ctx": args.num_ctx,
            "target_context_tokens": args.target_context_tokens,
            "long_prefix_repeat": long_prefix_repeat,
            "long_prefix_estimated_tokens": long_prefix_estimated_tokens,
            "long_prompt_estimated_tokens": _estimate_prompt_tokens(long_prompt),
            "keep_alive": args.keep_alive,
        },
        "short_chat": short_chat,
        "long_output_chat": long_output_chat,
        "long_prefix_run_1": long_prefix_run_1,
        "long_prefix_run_2": long_prefix_run_2,
        "multi_turn_chat": multi_turn_chat,
        "single_session_followup": single_session_followup,
        "single_session_recovery": single_session_recovery,
    }

    _print_section(
        "Summary",
        {
            "version": version,
            "installed_models": [item.get("name") for item in tags.get("models", []) if item.get("name")],
            "model": args.model,
            "think_enabled": args.think,
            "short_chat": {
                "wall_time_sec": short_chat["wall_time_sec"],
                "decode_speed": short_chat["decode_speed"],
                "answer_preview": short_chat["answer_preview"],
                "thinking_chars": short_chat["thinking_chars"],
            },
            "long_output_chat": {
                "wall_time_sec": long_output_chat["wall_time_sec"],
                "decode_speed": long_output_chat["decode_speed"],
                "thinking_chars": long_output_chat["thinking_chars"],
            },
            "long_prefix_run_1": {
                "wall_time_sec": long_prefix_run_1["wall_time_sec"],
                "decode_speed": long_prefix_run_1["decode_speed"],
                "thinking_chars": long_prefix_run_1["thinking_chars"],
            },
            "long_prefix_run_2": {
                "wall_time_sec": long_prefix_run_2["wall_time_sec"],
                "decode_speed": long_prefix_run_2["decode_speed"],
                "thinking_chars": long_prefix_run_2["thinking_chars"],
            },
            "single_session_followup": {
                "turn1_long_sec": single_session_followup["summary"]["turn1_long_sec"],
                "followup_avg_sec": single_session_followup["summary"]["followup_avg_sec"],
            },
            "single_session_recovery": {
                "turn1_cold_long_sec": single_session_recovery["summary"]["turn1_cold_long_sec"],
                "restart_wall_time_sec": single_session_recovery["summary"]["restart_wall_time_sec"],
                "ready_after_restart_sec": single_session_recovery["summary"]["ready_after_restart_sec"],
            },
        },
    )
    _print_section("Full Report", results)


if __name__ == "__main__":
    main()
