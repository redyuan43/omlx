#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Aggregate benchmark runs across multiple Qwen3.5-4B variants."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parent
OMLX_BENCH = ROOT / "bench_qwen35_4b.py"
OMLX_CONCURRENCY_BENCH = ROOT / "bench_qwen35_4b_concurrency.py"
LMSTUDIO_BENCH = ROOT / "bench_lmstudio_qwen35_4b.py"


def _run_json(command: List[str]) -> Dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    long_prefix_1 = payload.get("long_prefix_run_1", {})
    long_prefix_2 = payload.get("long_prefix_run_2", {})
    long_output = payload.get("long_output_chat", {})
    runtime_summary = payload.get("runtime_summary", {})
    summary = {
        "backend_format": runtime_summary.get("backend_format"),
        "quant_mode": runtime_summary.get("quant_mode"),
        "model_source": runtime_summary.get("model_source"),
        "gguf_variant": runtime_summary.get("gguf_variant"),
        "loaded_context_length": runtime_summary.get("loaded_context_length"),
        "ctx_size": runtime_summary.get("ctx_size"),
        "parallel_slots": runtime_summary.get("parallel_slots"),
        "short_chat_wall_time_sec": payload.get("short_chat", {}).get("wall_time_sec"),
        "multi_turn_wall_time_sec": payload.get("multi_turn_chat", {}).get("wall_time_sec"),
        "long_output_token_per_second": long_output.get("token_per_second"),
        "long_output_decode_token_per_second": long_output.get("decode_token_per_second"),
        "long_prefix_run_1_sec": long_prefix_1.get("wall_time_sec"),
        "long_prefix_run_2_sec": long_prefix_2.get("wall_time_sec"),
        "cached_prompt_tokens_run_2": long_prefix_2.get("cached_prompt_tokens"),
        "runtime_memory_after_long_prefix_run_2": long_prefix_2.get("runtime_memory_after"),
    }
    if long_prefix_1.get("wall_time_sec") and long_prefix_2.get("wall_time_sec"):
        summary["repeat_speedup_x"] = round(
            long_prefix_1["wall_time_sec"] / max(long_prefix_2["wall_time_sec"], 1e-6),
            2,
        )
    return summary


def _summarize_concurrency(payload: Dict[str, Any]) -> Dict[str, Any]:
    repeat = payload.get("repeat_long_prefix_plus_short", {})
    parallel = repeat.get("parallel", {})
    return {
        "dual_short_makespan_sec": payload.get("dual_short_independent", {}).get("makespan_sec"),
        "dual_long_output_makespan_sec": payload.get("dual_long_output_independent", {}).get("makespan_sec"),
        "dual_long_output_aggregate_token_per_second": payload.get("dual_long_output_independent", {}).get(
            "aggregate_token_per_second"
        ),
        "dual_long_prefix_makespan_sec": payload.get("dual_long_prefix_independent", {}).get("makespan_sec"),
        "repeat_parallel_makespan_sec": parallel.get("makespan_sec"),
        "repeat_speedup_x": repeat.get("repeat_speedup_x"),
    }


def _parse_omlx_variant(raw_value: str) -> tuple[str, str, str]:
    try:
        label, control_plane_url, runtime_url = raw_value.split(",", 2)
    except ValueError as exc:
        raise SystemExit(
            f"invalid --omlx-variant '{raw_value}', expected LABEL,CONTROL_PLANE_URL,RUNTIME_URL"
        ) from exc
    return label, control_plane_url, runtime_url


def _parse_lmstudio_variant(raw_value: str) -> tuple[str, str, str]:
    try:
        label, lmstudio_url, model = raw_value.split(",", 2)
    except ValueError as exc:
        raise SystemExit(
            f"invalid --lmstudio-variant '{raw_value}', expected LABEL,LMSTUDIO_URL,MODEL"
        ) from exc
    return label, lmstudio_url, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multiple Qwen3.5-4B benchmark variants")
    parser.add_argument(
        "--omlx-variant",
        action="append",
        default=[],
        help="LABEL,CONTROL_PLANE_URL,RUNTIME_URL",
    )
    parser.add_argument(
        "--lmstudio-variant",
        action="append",
        default=[],
        help="LABEL,LMSTUDIO_URL,MODEL",
    )
    parser.add_argument("--long-prefix-repeat", type=int, default=300)
    parser.add_argument("--long-output-max-tokens", type=int, default=192)
    parser.add_argument("--prefix-salt", default="")
    parser.add_argument("--include-concurrency", action="store_true")
    args = parser.parse_args()

    variants: Dict[str, Dict[str, Any]] = {}
    concurrency_variants: Dict[str, Dict[str, Any]] = {}
    for raw_variant in args.omlx_variant:
        label, control_plane_url, runtime_url = _parse_omlx_variant(raw_variant)
        command = [
            sys.executable,
            str(OMLX_BENCH),
            "--control-plane-url",
            control_plane_url,
            "--runtime-url",
            runtime_url,
            "--long-prefix-repeat",
            str(args.long_prefix_repeat),
            "--long-output-max-tokens",
            str(args.long_output_max_tokens),
        ]
        if args.prefix_salt:
            command.extend(["--prefix-salt", f"{args.prefix_salt}-{label}"])
        variants[label] = _run_json(command)
        if args.include_concurrency:
            concurrency_command = [
                sys.executable,
                str(OMLX_CONCURRENCY_BENCH),
                "--control-plane-url",
                control_plane_url,
                "--runtime-url",
                runtime_url,
                "--long-prefix-repeat",
                str(args.long_prefix_repeat),
                "--long-output-max-tokens",
                str(args.long_output_max_tokens),
            ]
            if args.prefix_salt:
                concurrency_command.extend(
                    ["--prefix-salt", f"{args.prefix_salt}-{label}-concurrency"]
                )
            concurrency_variants[label] = _run_json(concurrency_command)

    for raw_variant in args.lmstudio_variant:
        label, lmstudio_url, model = _parse_lmstudio_variant(raw_variant)
        command = [
            sys.executable,
            str(LMSTUDIO_BENCH),
            "--lmstudio-url",
            lmstudio_url,
            "--model",
            model,
            "--long-prefix-repeat",
            str(args.long_prefix_repeat),
            "--long-output-max-tokens",
            str(args.long_output_max_tokens),
        ]
        if args.prefix_salt:
            command.extend(["--prefix-salt", f"{args.prefix_salt}-{label}"])
        variants[label] = _run_json(command)

    output = {
        "variants": variants,
        "summary": {
            label: _summarize(payload) for label, payload in variants.items()
        },
    }
    if concurrency_variants:
        output["concurrency"] = concurrency_variants
        output["concurrency_summary"] = {
            label: _summarize_concurrency(payload)
            for label, payload in concurrency_variants.items()
        }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
