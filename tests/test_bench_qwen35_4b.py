# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_bench_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "bench_qwen35_4b.py"
    spec = importlib.util.spec_from_file_location("bench_qwen35_4b", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench_qwen35_4b = _load_bench_module()


def test_build_incremental_suffix_uses_exact_single_token_repeat(monkeypatch):
    def fake_tokenize(runtime_url: str, text: str):
        if text == "BASE":
            return [0] * 100
        if text == " a":
            return [264]
        if text == " b":
            return [265]
        if text.startswith("BASE"):
            suffix = text[len("BASE") :]
            if suffix == " a" * 78:
                return [0] * 178
            if suffix == " b" * 78:
                return [0] * 160
        return None

    def fake_detokenize(runtime_url: str, tokens: list[int]):
        if tokens and all(token == 264 for token in tokens):
            return " a" * len(tokens)
        if tokens and all(token == 265 for token in tokens):
            return " b" * len(tokens)
        return None

    monkeypatch.setattr(bench_qwen35_4b, "_tokenize_content", fake_tokenize)
    monkeypatch.setattr(bench_qwen35_4b, "_detokenize_tokens", fake_detokenize)

    delta = bench_qwen35_4b._build_incremental_suffix("http://runtime", "BASE", 78)

    assert delta["mode"] == "exact_repeat_single_token"
    assert delta["fragment"] == " a"
    assert delta["token_id"] == 264
    assert delta["base_prompt_tokens"] == 100
    assert delta["actual_delta_tokens"] == 78
    assert delta["suffix"] == " a" * 78


def test_run_incremental_long_prefix_summarizes_prompt_delta(monkeypatch):
    monkeypatch.setattr(
        bench_qwen35_4b,
        "_build_incremental_suffix",
        lambda runtime_url, base_prompt, target_delta_tokens: {
            "suffix": " a" * 78,
            "mode": "exact_repeat_single_token",
            "fragment": " a",
            "token_id": 264,
            "base_prompt_tokens": 1000,
            "actual_delta_tokens": 78,
        },
    )

    responses = iter(
        [
            {
                "wall_time_sec": 60.705,
                "usage": {"prompt_tokens": 57830},
                "cached_prompt_tokens": 0,
                "continuation_hit": False,
                "prefix_drift": None,
            },
            {
                "wall_time_sec": 3.685,
                "usage": {"prompt_tokens": 57908},
                "cached_prompt_tokens": 57826,
                "continuation_hit": True,
                "prefix_drift": 0,
            },
        ]
    )
    monkeypatch.setattr(bench_qwen35_4b, "_run_completion_case", lambda **kwargs: next(responses))

    result = bench_qwen35_4b._run_incremental_long_prefix(
        control_plane_url="http://127.0.0.1:8008",
        runtime_url="http://127.0.0.1:30000",
        model="qwen35",
        base_prompt="LONG",
        max_tokens=4,
        conversation_id="bench-incremental-long-prefix",
        target_delta_tokens=78,
    )

    assert result["delta"]["target_tokens"] == 78
    assert result["delta"]["actual_delta_tokens"] == 78
    assert result["delta"]["prompt_tokens_delta_vs_run1"] == 78
    assert result["summary"]["run2_cached_prompt_tokens"] == 57826
    assert result["summary"]["run2_continuation_hit"] is True
    assert result["summary"]["prompt_tokens_delta_vs_run1"] == 78
    assert result["summary"]["speedup_x"] == 16.47
