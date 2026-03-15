#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the local LM Studio Qwen3.5-4B GGUF model."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict


def _http_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def _read_meminfo() -> Dict[str, int]:
    keys = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, _, raw_value = line.partition(":")
        if name not in keys:
            continue
        values[name] = int(raw_value.strip().split()[0])
    return values


def _lookup_model(lmstudio_url: str, model: str) -> Dict[str, Any]:
    payload = _http_json(f"{lmstudio_url}/api/v0/models")
    for item in payload.get("data", []):
        if item.get("id") == model:
            return item
    return {}


def _runtime_summary(model_details: Dict[str, Any]) -> Dict[str, Any]:
    model_path = model_details.get("path") or ""
    match = re.search(r"(IQ\d+_[A-Z0-9_]+|Q\d+_[A-Z0-9_]+)", Path(model_path).name, re.IGNORECASE)
    return {
        "backend_format": "lmstudio_gguf",
        "quant_mode": "lmstudio_baseline",
        "model_source": "lmstudio_api",
        "loaded_context_length": model_details.get("loaded_context_length"),
        "compatibility_type": model_details.get("compatibility_type"),
        "quantization": model_details.get("quantization"),
        "gguf_variant": (match.group(1).upper() if match else model_details.get("quantization")),
        "publisher": model_details.get("publisher"),
        "path": model_path,
    }


def _run_case(
    *,
    lmstudio_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> Dict[str, Any]:
    before = _read_meminfo()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "enableThinking": False,
        "reasoning": False,
    }

    started = time.perf_counter()
    response = _http_json(
        f"{lmstudio_url}/v1/chat/completions",
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
        "assistant": response["choices"][0]["message"],
        "system_mem_before_kib": before,
        "system_mem_after_kib": _read_meminfo(),
    }


def _run_messages_case(
    *,
    lmstudio_url: str,
    model: str,
    messages: list[Dict[str, Any]],
    max_tokens: int,
) -> Dict[str, Any]:
    before = _read_meminfo()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "enableThinking": False,
        "reasoning": False,
    }

    started = time.perf_counter()
    response = _http_json(
        f"{lmstudio_url}/v1/chat/completions",
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
        "assistant": response["choices"][0]["message"],
        "system_mem_before_kib": before,
        "system_mem_after_kib": _read_meminfo(),
    }


def _run_single_session_followup(
    *,
    lmstudio_url: str,
    model: str,
    long_prompt: str,
) -> Dict[str, Any]:
    turn1_messages = [
        {
            "role": "user",
            "content": f"{long_prompt}\n\nExplain in one short sentence why reusing context matters.",
        }
    ]
    turn1 = _run_messages_case(
        lmstudio_url=lmstudio_url,
        model=model,
        messages=turn1_messages,
        max_tokens=24,
    )
    turn2_messages = turn1_messages + [
        turn1["assistant"],
        {"role": "user", "content": "Now answer with exactly one word: benefit?"},
    ]
    turn2 = _run_messages_case(
        lmstudio_url=lmstudio_url,
        model=model,
        messages=turn2_messages,
        max_tokens=8,
    )
    turn3_messages = turn2_messages + [
        turn2["assistant"],
        {"role": "user", "content": "Again, one different word only."},
    ]
    turn3 = _run_messages_case(
        lmstudio_url=lmstudio_url,
        model=model,
        messages=turn3_messages,
        max_tokens=8,
    )
    followup_avg = round((turn2["wall_time_sec"] + turn3["wall_time_sec"]) / 2, 3)
    return {
        "turn1_long": turn1,
        "turn2_short_followup": turn2,
        "turn3_short_followup": turn3,
        "summary": {
            "turn1_long_sec": turn1.get("wall_time_sec"),
            "turn2_short_followup_sec": turn2.get("wall_time_sec"),
            "turn3_short_followup_sec": turn3.get("wall_time_sec"),
            "turn2_prompt_tokens": (turn2.get("usage") or {}).get("prompt_tokens"),
            "turn3_prompt_tokens": (turn3.get("usage") or {}).get("prompt_tokens"),
            "followup_avg_sec": followup_avg,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LM Studio Qwen3.5-4B GGUF")
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="qwen3.5-4b")
    parser.add_argument("--long-prefix-repeat", type=int, default=300)
    parser.add_argument("--long-output-max-tokens", type=int, default=192)
    parser.add_argument("--prefix-salt", default="")
    args = parser.parse_args()

    prefix_unit = "LM Studio Qwen3.5 cache benchmark"
    if args.prefix_salt:
        prefix_unit = f"{prefix_unit} {args.prefix_salt}"
    long_prefix = " ".join([prefix_unit] * args.long_prefix_repeat)
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

    models = _http_json(f"{args.lmstudio_url}/v1/models")
    model_details = _lookup_model(args.lmstudio_url, args.model)
    results = {
        "models": models,
        "model_details": model_details,
        "runtime_summary": _runtime_summary(model_details),
        "short_chat": _run_case(
            lmstudio_url=args.lmstudio_url,
            model=args.model,
            prompt="Reply with exactly PONG.",
            max_tokens=16,
        ),
        "long_output_chat": _run_case(
            lmstudio_url=args.lmstudio_url,
            model=args.model,
            prompt=long_output_prompt,
            max_tokens=args.long_output_max_tokens,
        ),
        "long_prefix_run_1": _run_case(
            lmstudio_url=args.lmstudio_url,
            model=args.model,
            prompt=long_prompt,
            max_tokens=4,
        ),
        "long_prefix_run_2": _run_case(
            lmstudio_url=args.lmstudio_url,
            model=args.model,
            prompt=long_prompt,
            max_tokens=4,
        ),
        "multi_turn_chat": _run_messages_case(
            lmstudio_url=args.lmstudio_url,
            model=args.model,
            messages=multi_turn_messages,
            max_tokens=64,
        ),
        "single_session_followup": _run_single_session_followup(
            lmstudio_url=args.lmstudio_url,
            model=args.model,
            long_prompt=long_prefix,
        ),
    }
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
