#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark the current Qwen3.5-4B control-plane/runtime stack."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict


def _http_json(url: str, *, method: str = "GET", payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    last_error = None
    for attempt in range(4):
        request = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
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


def _http_text(url: str) -> str:
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def _read_meminfo() -> Dict[str, int]:
    keys = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values: Dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, _, raw_value = line.partition(":")
        if name not in keys:
            continue
        values[name] = int(raw_value.strip().split()[0])
    return values


def _extract_metric(metrics_text: str, metric_name: str) -> float | None:
    pattern = re.compile(
        rf"^{re.escape(metric_name)}(?:\{{[^}}]*\}})?\s+([0-9eE+.-]+)$",
        re.MULTILINE,
    )
    match = pattern.search(metrics_text)
    if not match:
        return None
    return float(match.group(1))


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


def _runtime_memory_usage(runtime_payload: Dict[str, Any]) -> Dict[str, Any]:
    backend = runtime_payload.get("backend", {})
    server_info = (
        backend
        .get("details", {})
        .get("server_info", {})
    )
    memory_usage = server_info.get("memory_usage")
    if isinstance(memory_usage, dict) and memory_usage:
        return dict(memory_usage)

    internal_states = server_info.get("internal_states", [])
    if internal_states and isinstance(internal_states[0], dict):
        nested = internal_states[0].get("memory_usage")
        if isinstance(nested, dict):
            return dict(nested)
    fallback = {}
    for key in ("gpu_memory_used_mb", "gpu_memory_total_mb", "gpu_util_percent"):
        value = backend.get(key)
        if value is not None:
            fallback[key] = value
    if fallback:
        return fallback
    return {}


def _runtime_summary(runtime_payload: Dict[str, Any]) -> Dict[str, Any]:
    backend = runtime_payload.get("backend", {})
    details = backend.get("details", {})
    diagnostics = details.get("diagnostics", {})
    telemetry = details.get("telemetry", {})
    config_backend = runtime_payload.get("config", {}).get("backend", {})
    routing = details.get("routing", {})
    props = details.get("props", {})
    server_info = (
        backend
        .get("details", {})
        .get("server_info", {})
    )
    internal_states = server_info.get("internal_states", [])
    state = internal_states[0] if internal_states and isinstance(internal_states[0], dict) else {}
    load_format = server_info.get("load_format")
    quantization = server_info.get("quantization")
    inferred_quant_mode = "bf16"
    inferred_backend_format = "sglang_bf16"
    inferred_model_source = "hf"
    if load_format == "gguf" or quantization == "gguf":
        inferred_quant_mode = "gguf_experimental"
        inferred_backend_format = "sglang_gguf_experimental"
        inferred_model_source = "gguf"
    elif quantization == "awq":
        inferred_quant_mode = "awq_int4"
        inferred_backend_format = "sglang_awq_int4"
    elif diagnostics.get("adapter") == "llama_cpp" or config_backend.get("kind") == "llama_cpp":
        inferred_quant_mode = "gguf_experimental"
        inferred_backend_format = "llama_cpp_gguf"
        inferred_model_source = "gguf"
    return {
        "backend_format": (
            diagnostics.get("backend_format")
            or config_backend.get("backend_format")
            or inferred_backend_format
        ),
        "quant_mode": (
            diagnostics.get("quant_mode")
            or config_backend.get("quant_mode")
            or inferred_quant_mode
        ),
        "model_source": (
            diagnostics.get("model_source")
            or config_backend.get("model_source")
            or inferred_model_source
        ),
        "loaded_context_length": (
            state.get("context_length")
            or server_info.get("context_length")
            or props.get("default_generation_settings", {}).get("n_ctx")
            or props.get("n_ctx")
            or diagnostics.get("ctx_size")
            or config_backend.get("ctx_size")
            or config_backend.get("context_length")
        ),
        "effective_model_path": diagnostics.get("effective_model_path")
        or props.get("model_path")
        or server_info.get("model_path"),
        "artifact_summary": diagnostics.get("artifact_summary"),
        "serving_preset": diagnostics.get("serving_preset")
        or config_backend.get("serving_preset"),
        "gguf_variant": diagnostics.get("gguf_variant")
        or (diagnostics.get("artifact_summary") or {}).get("gguf_variant"),
        "chunked_prefill_size": state.get(
            "chunked_prefill_size", server_info.get("chunked_prefill_size")
        ),
        "ctx_size": diagnostics.get("ctx_size") or config_backend.get("ctx_size"),
        "parallel_slots": diagnostics.get("parallel_slots") or config_backend.get("parallel_slots"),
        "n_gpu_layers": diagnostics.get("n_gpu_layers") or config_backend.get("n_gpu_layers"),
        "flash_attn": diagnostics.get("flash_attn", config_backend.get("flash_attn")),
        "batch_size": diagnostics.get("batch_size") or config_backend.get("batch_size"),
        "ubatch_size": diagnostics.get("ubatch_size") or config_backend.get("ubatch_size"),
        "cache_ram_mib": diagnostics.get("cache_ram_mib") or config_backend.get("cache_ram_mib"),
        "cache_reuse": diagnostics.get("cache_reuse") or config_backend.get("cache_reuse"),
        "checkpoint_every_n_tokens": diagnostics.get("checkpoint_every_n_tokens")
        or config_backend.get("checkpoint_every_n_tokens"),
        "ctx_checkpoints": diagnostics.get("ctx_checkpoints")
        or config_backend.get("ctx_checkpoints"),
        "slot_prompt_similarity": diagnostics.get("slot_prompt_similarity")
        or config_backend.get("slot_prompt_similarity"),
        "enable_runtime_metrics": diagnostics.get("enable_runtime_metrics")
        if diagnostics.get("enable_runtime_metrics") is not None
        else config_backend.get("enable_runtime_metrics"),
        "enable_session_stickiness": diagnostics.get("enable_session_stickiness")
        if diagnostics.get("enable_session_stickiness") is not None
        else config_backend.get("enable_session_stickiness"),
        "single_session_continuation_enabled": diagnostics.get(
            "single_session_continuation_enabled"
        ),
        "single_session_continuation_ttl_seconds": diagnostics.get(
            "single_session_continuation_ttl_seconds"
        ),
        "sticky_session_prompt_threshold": diagnostics.get("sticky_session_prompt_threshold")
        or config_backend.get("sticky_session_prompt_threshold"),
        "sticky_max_sessions": diagnostics.get("sticky_max_sessions")
        or config_backend.get("sticky_max_sessions"),
        "split_mode": diagnostics.get("split_mode") or config_backend.get("split_mode"),
        "no_context_shift": diagnostics.get(
            "no_context_shift", config_backend.get("no_context_shift")
        ),
        "enable_hierarchical_cache": state.get(
            "enable_hierarchical_cache", server_info.get("enable_hierarchical_cache")
        ),
        "disable_chunked_prefix_cache": state.get(
            "disable_chunked_prefix_cache", server_info.get("disable_chunked_prefix_cache")
        ),
        "memory_usage": _runtime_memory_usage(runtime_payload),
        "telemetry": telemetry,
        "prefill_strategy": runtime_payload.get("config", {})
        .get("backend", {})
        .get("prefill_strategy", "fixed"),
        "routing_profiles": routing.get("profiles"),
        "last_route": routing.get("last_decision"),
        "continuation": details.get("continuation"),
        "session_restore": details.get("session_restore"),
    }


def _snapshot(control_plane_url: str, runtime_url: str) -> Dict[str, Any]:
    runtime = _http_json(f"{control_plane_url}/admin/api/runtime")
    if runtime_url:
        try:
            metrics_text = _http_text(f"{runtime_url}/metrics")
        except Exception:
            metrics_text = ""
    else:
        metrics_text = ""
    return {
        "meminfo": _read_meminfo(),
        "runtime": runtime,
        "metrics_text": metrics_text,
    }


def _delta_metric(before: str, after: str, sum_name: str, count_name: str) -> float | None:
    before_sum = _extract_metric(before, sum_name) or 0.0
    after_sum = _extract_metric(after, sum_name) or 0.0
    before_count = _extract_metric(before, count_name) or 0.0
    after_count = _extract_metric(after, count_name) or 0.0
    count_delta = after_count - before_count
    if count_delta <= 0:
        return None
    return (after_sum - before_sum) / count_delta


def _run_case(
    *,
    control_plane_url: str,
    runtime_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    disable_thinking: bool,
    llama_cpp_reasoning_compat: bool = False,
    conversation_id: str | None = None,
) -> Dict[str, Any]:
    before = _snapshot(control_plane_url, runtime_url)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if conversation_id:
        payload["metadata"] = {"conversation_id": conversation_id}
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        if llama_cpp_reasoning_compat:
            payload["enableThinking"] = False
            payload["reasoning"] = False
            payload["reasoning_budget"] = 0
            payload["reasoning_format"] = "none"
            payload["thinking_forced_open"] = False

    started = time.perf_counter()
    try:
        response = _http_json(
            f"{control_plane_url}/v1/chat/completions",
            method="POST",
            payload=payload,
        )
        error = None
    except urllib.error.HTTPError as exc:
        response = None
        error = {
            "status_code": exc.code,
            "body": exc.read().decode("utf-8", errors="replace"),
        }
    wall_time = time.perf_counter() - started
    after = _snapshot(control_plane_url, runtime_url)

    if response is None:
        continuation_after = (
            after["runtime"]
            .get("backend", {})
            .get("details", {})
            .get("continuation", {})
            .get("last_decision")
        )
        session_restore_after = (
            after["runtime"]
            .get("backend", {})
            .get("details", {})
            .get("session_restore", {})
        )
        return {
            "wall_time_sec": round(wall_time, 3),
            "ttft_sec": None,
            "e2e_sec": None,
            "token_per_second": None,
            "decode_token_per_second": None,
            "error": error,
            "system_mem_before_kib": before["meminfo"],
            "system_mem_after_kib": after["meminfo"],
            "runtime_memory_before": _runtime_memory_usage(before["runtime"]),
            "runtime_memory_after": _runtime_memory_usage(after["runtime"]),
            "cache_hit_rate_after": _extract_metric(
                after["metrics_text"], "sglang:cache_hit_rate"
            ),
            "num_used_tokens_after": _extract_metric(
                after["metrics_text"], "sglang:num_used_tokens"
            ),
            "route_after": (
                after["runtime"]
                .get("backend", {})
                .get("details", {})
                .get("routing", {})
                .get("last_decision")
            ),
            "slot_after": (
                after["runtime"]
                .get("backend", {})
                .get("details", {})
                .get("slot_router", {})
                .get("last_decision")
            ),
            "continuation_after": continuation_after,
            "session_restore_after": session_restore_after,
            "continuation_hit": (
                None if continuation_after is None else continuation_after.get("continuation_hit")
            ),
            "prefix_drift": (
                None if continuation_after is None else continuation_after.get("prefix_drift")
            ),
        }

    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0) or 0
    ttft = _delta_metric(
        before["metrics_text"],
        after["metrics_text"],
        "sglang:time_to_first_token_seconds_sum",
        "sglang:time_to_first_token_seconds_count",
    )
    e2e = _delta_metric(
        before["metrics_text"],
        after["metrics_text"],
        "sglang:e2e_request_latency_seconds_sum",
        "sglang:e2e_request_latency_seconds_count",
    )
    token_per_second = (
        completion_tokens / max(wall_time, 1e-6) if completion_tokens else None
    )
    decode_window = None
    decode_token_per_second = None
    if completion_tokens and e2e is not None and ttft is not None and e2e > ttft:
        decode_window = e2e - ttft
        decode_token_per_second = completion_tokens / decode_window
    continuation_after = (
        after["runtime"]
        .get("backend", {})
        .get("details", {})
        .get("continuation", {})
        .get("last_decision")
    )
    session_restore_after = (
        after["runtime"]
        .get("backend", {})
        .get("details", {})
        .get("session_restore", {})
    )

    return {
        "wall_time_sec": round(wall_time, 3),
        "ttft_sec": None if ttft is None else round(ttft, 3),
        "e2e_sec": None if e2e is None else round(e2e, 3),
        "token_per_second": None if token_per_second is None else round(token_per_second, 3),
        "decode_token_per_second": (
            None if decode_token_per_second is None else round(decode_token_per_second, 3)
        ),
        "usage": usage,
        "cached_prompt_tokens": (
            usage.get("prompt_tokens_details", {}) or {}
        ).get("cached_tokens"),
        "assistant": response["choices"][0]["message"],
        "system_mem_before_kib": before["meminfo"],
        "system_mem_after_kib": after["meminfo"],
        "runtime_memory_before": _runtime_memory_usage(before["runtime"]),
        "runtime_memory_after": _runtime_memory_usage(after["runtime"]),
        "cache_hit_rate_after": _extract_metric(
            after["metrics_text"], "sglang:cache_hit_rate"
        ),
        "num_used_tokens_after": _extract_metric(
            after["metrics_text"], "sglang:num_used_tokens"
        ),
        "route_after": (
            after["runtime"]
            .get("backend", {})
            .get("details", {})
            .get("routing", {})
            .get("last_decision")
        ),
        "slot_after": (
            after["runtime"]
            .get("backend", {})
            .get("details", {})
            .get("slot_router", {})
            .get("last_decision")
        ),
        "continuation_after": continuation_after,
        "session_restore_after": session_restore_after,
        "continuation_hit": (
            None if continuation_after is None else continuation_after.get("continuation_hit")
        ),
        "prefix_drift": (
            None if continuation_after is None else continuation_after.get("prefix_drift")
        ),
    }


def _run_messages_case(
    *,
    control_plane_url: str,
    runtime_url: str,
    model: str,
    messages: list[Dict[str, Any]],
    max_tokens: int,
    disable_thinking: bool,
    llama_cpp_reasoning_compat: bool = False,
    conversation_id: str | None = None,
) -> Dict[str, Any]:
    before = _snapshot(control_plane_url, runtime_url)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if conversation_id:
        payload["metadata"] = {"conversation_id": conversation_id}
    if disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        if llama_cpp_reasoning_compat:
            payload["enableThinking"] = False
            payload["reasoning"] = False
            payload["reasoning_budget"] = 0
            payload["reasoning_format"] = "none"
            payload["thinking_forced_open"] = False

    started = time.perf_counter()
    response = _http_json(
        f"{control_plane_url}/v1/chat/completions",
        method="POST",
        payload=payload,
    )
    wall_time = time.perf_counter() - started
    after = _snapshot(control_plane_url, runtime_url)
    usage = response.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0) or 0
    continuation_after = (
        after["runtime"]
        .get("backend", {})
        .get("details", {})
        .get("continuation", {})
        .get("last_decision")
    )
    session_restore_after = (
        after["runtime"]
        .get("backend", {})
        .get("details", {})
        .get("session_restore", {})
    )
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
        "system_mem_before_kib": before["meminfo"],
        "system_mem_after_kib": after["meminfo"],
        "runtime_memory_before": _runtime_memory_usage(before["runtime"]),
        "runtime_memory_after": _runtime_memory_usage(after["runtime"]),
        "route_after": (
            after["runtime"]
            .get("backend", {})
            .get("details", {})
            .get("routing", {})
            .get("last_decision")
        ),
        "slot_after": (
            after["runtime"]
            .get("backend", {})
            .get("details", {})
            .get("slot_router", {})
            .get("last_decision")
        ),
        "continuation_after": continuation_after,
        "session_restore_after": session_restore_after,
        "continuation_hit": (
            None if continuation_after is None else continuation_after.get("continuation_hit")
        ),
        "prefix_drift": (
            None if continuation_after is None else continuation_after.get("prefix_drift")
        ),
    }


def _run_single_session_followup(
    *,
    control_plane_url: str,
    runtime_url: str,
    model: str,
    long_prompt: str,
    disable_thinking: bool,
    llama_cpp_reasoning_compat: bool,
    conversation_id: str,
) -> Dict[str, Any]:
    turn1_messages = [
        {
            "role": "user",
            "content": f"{long_prompt}\n\nExplain in one short sentence why reusing context matters.",
        }
    ]
    turn1 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn1_messages,
        max_tokens=24,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )
    turn2_messages = turn1_messages + [
        turn1["assistant"],
        {"role": "user", "content": "Now answer with exactly one word: benefit?"},
    ]
    turn2 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn2_messages,
        max_tokens=8,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )
    turn3_messages = turn2_messages + [
        turn2["assistant"],
        {"role": "user", "content": "Again, one different word only."},
    ]
    turn3 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn3_messages,
        max_tokens=8,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )
    turn2_prompt_tokens = (turn2.get("usage") or {}).get("prompt_tokens")
    turn3_prompt_tokens = (turn3.get("usage") or {}).get("prompt_tokens")
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
            "turn2_prompt_tokens": turn2_prompt_tokens,
            "turn3_prompt_tokens": turn3_prompt_tokens,
            "turn2_continuation_hit": turn2.get("continuation_hit"),
            "turn3_continuation_hit": turn3.get("continuation_hit"),
            "turn2_prefix_drift": turn2.get("prefix_drift"),
            "turn3_prefix_drift": turn3.get("prefix_drift"),
            "followup_avg_sec": followup_avg,
            "slot_after": turn3.get("slot_after"),
            "continuation_after": turn3.get("continuation_after"),
        },
    }


def _restart_managed_runtime(control_plane_url: str) -> Dict[str, Any]:
    stop_result = _http_json(
        f"{control_plane_url}/admin/api/runtime/stop",
        method="POST",
        payload={},
    )
    start_result = _http_json(
        f"{control_plane_url}/admin/api/runtime/start",
        method="POST",
        payload={},
    )
    return {
        "stop": stop_result,
        "start": start_result,
    }


def _run_single_session_recovery(
    *,
    control_plane_url: str,
    runtime_url: str,
    model: str,
    long_prompt: str,
    disable_thinking: bool,
    llama_cpp_reasoning_compat: bool,
    conversation_id: str,
    runtime_summary: Dict[str, Any],
    initial_runtime: Dict[str, Any],
) -> Dict[str, Any]:
    if runtime_summary.get("backend_format") != "llama_cpp_gguf":
        return {"skipped": True, "reason": "backend_not_llama_cpp"}
    diagnostics = (
        initial_runtime.get("backend", {})
        .get("details", {})
        .get("diagnostics", {})
    )
    if not diagnostics.get("managed_process_running"):
        return {"skipped": True, "reason": "runtime_not_managed_by_control_plane"}
    if not runtime_summary.get("single_session_continuation_enabled"):
        return {"skipped": True, "reason": "single_session_continuation_disabled"}

    turn1_messages = [
        {
            "role": "user",
            "content": (
                f"{long_prompt}\n\n"
                "Explain in one short sentence why reusing context matters."
            ),
        }
    ]
    turn1 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn1_messages,
        max_tokens=24,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )
    turn2_messages = turn1_messages + [
        turn1["assistant"],
        {"role": "user", "content": "Now answer with exactly one word: benefit?"},
    ]
    turn2 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn2_messages,
        max_tokens=8,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )

    restart = _restart_managed_runtime(control_plane_url)

    turn3_messages = turn2_messages + [
        turn2["assistant"],
        {
            "role": "user",
            "content": "After the short runtime restart, answer with one different word.",
        },
    ]
    turn3 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn3_messages,
        max_tokens=8,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )
    turn4_messages = turn3_messages + [
        turn3["assistant"],
        {"role": "user", "content": "Again, one different word only."},
    ]
    turn4 = _run_messages_case(
        control_plane_url=control_plane_url,
        runtime_url=runtime_url,
        model=model,
        messages=turn4_messages,
        max_tokens=8,
        disable_thinking=disable_thinking,
        llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
        conversation_id=conversation_id,
    )
    turn3_restore = turn3.get("session_restore_after") or {}
    turn3_last_restore = (
        turn3_restore.get("last_restore")
        if isinstance(turn3_restore, dict)
        else None
    )
    cold_to_restored_speedup = None
    if turn1.get("wall_time_sec") and turn3.get("wall_time_sec"):
        cold_to_restored_speedup = round(
            turn1["wall_time_sec"] / max(turn3["wall_time_sec"], 1e-6),
            2,
        )
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
            "turn3_recovery_replay_sec": turn3.get("wall_time_sec"),
            "turn4_post_recovery_followup_sec": turn4.get("wall_time_sec"),
            "turn2_continuation_hit": turn2.get("continuation_hit"),
            "turn3_continuation_hit": turn3.get("continuation_hit"),
            "turn4_continuation_hit": turn4.get("continuation_hit"),
            "turn3_restore_ok": (
                None
                if not isinstance(turn3_last_restore, dict)
                else turn3_last_restore.get("ok")
            ),
            "turn3_restore_status": (
                None
                if not isinstance(turn3_last_restore, dict)
                else turn3_last_restore.get("status")
            ),
            "turn3_restore_ms": (
                None
                if not isinstance(turn3_last_restore, dict)
                else turn3_last_restore.get("restore_ms")
            ),
            "turn3_restore_n_restored": (
                None
                if not isinstance(turn3_last_restore, dict)
                else turn3_last_restore.get("n_restored")
            ),
            "turn3_recovery_reason": (
                (turn3.get("continuation_after") or {}).get("reason")
            ),
            "turn4_post_recovery_reason": (
                (turn4.get("continuation_after") or {}).get("reason")
            ),
            "cold_to_restored_speedup_x": cold_to_restored_speedup,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Qwen3.5-4B on the local control plane")
    parser.add_argument("--control-plane-url", default="http://127.0.0.1:8010")
    parser.add_argument("--runtime-url", default="http://127.0.0.1:31000")
    parser.add_argument("--model", default="qwen35-4b")
    parser.add_argument("--long-prefix-repeat", type=int, default=0)
    parser.add_argument("--long-output-max-tokens", type=int, default=192)
    parser.add_argument("--target-context-tokens", type=int, default=65536)
    parser.add_argument("--prefix-salt", default="")
    parser.add_argument("--disable-thinking", action="store_true", default=True)
    args = parser.parse_args()

    short_prompt = "Reply with exactly PONG."
    prefix_unit = "DGX Spark Qwen3.5 cache benchmark"
    if args.prefix_salt:
        prefix_unit = f"{prefix_unit} {args.prefix_salt}"
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

    try:
        health = _http_json(f"{args.control_plane_url}/health")
    except urllib.error.URLError as exc:
        raise SystemExit(f"control plane unavailable: {exc}") from exc

    initial_runtime = _http_json(f"{args.control_plane_url}/admin/api/runtime")
    runtime_summary = _runtime_summary(initial_runtime)
    llama_cpp_reasoning_compat = runtime_summary.get("backend_format") == "llama_cpp_gguf"
    loaded_context_length = runtime_summary.get("loaded_context_length")
    if isinstance(loaded_context_length, int) and loaded_context_length > 0:
        effective_context_tokens = min(args.target_context_tokens, loaded_context_length)
    else:
        effective_context_tokens = args.target_context_tokens
    if args.long_prefix_repeat > 0:
        long_prefix = " ".join([prefix_unit] * args.long_prefix_repeat)
        long_prefix_repeat = args.long_prefix_repeat
        long_prefix_estimated_tokens = _estimate_prompt_tokens(long_prefix)
    else:
        context_headroom = max(
            4096,
            min(16384, effective_context_tokens // 4),
        )
        long_prefix_budget = max(
            1024,
            effective_context_tokens - context_headroom,
        )
        long_prefix, long_prefix_repeat, long_prefix_estimated_tokens = _build_long_prefix(
            prefix_unit,
            long_prefix_budget,
        )
    long_prompt = f"{long_prefix}\n\nReply with exactly OK."
    long_prompt_estimated_tokens = _estimate_prompt_tokens(long_prompt)

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
            "long_prompt_estimated_tokens": long_prompt_estimated_tokens,
        },
        "short_chat": _run_case(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            prompt=short_prompt,
            max_tokens=16,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-short-chat",
        ),
        "long_output_chat": _run_case(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            prompt=long_output_prompt,
            max_tokens=args.long_output_max_tokens,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-long-output",
        ),
        "long_prefix_run_1": _run_case(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            prompt=long_prompt,
            max_tokens=4,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-repeat-long-prefix",
        ),
        "long_prefix_run_2": _run_case(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            prompt=long_prompt,
            max_tokens=4,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-repeat-long-prefix",
        ),
        "multi_turn_chat": _run_messages_case(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            messages=multi_turn_messages,
            max_tokens=64,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-multi-turn",
        ),
        "single_session_followup": _run_single_session_followup(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            long_prompt=long_prefix,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-single-session-followup",
        ),
        "single_session_recovery": _run_single_session_recovery(
            control_plane_url=args.control_plane_url,
            runtime_url=args.runtime_url,
            model=args.model,
            long_prompt=long_prefix,
            disable_thinking=args.disable_thinking,
            llama_cpp_reasoning_compat=llama_cpp_reasoning_compat,
            conversation_id="bench-single-session-recovery",
            runtime_summary=runtime_summary,
            initial_runtime=initial_runtime,
        ),
    }

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
