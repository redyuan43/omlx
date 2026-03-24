# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mixed-traffic concurrency benchmark sizing."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import bench_qwen35_4b_concurrency as concurrency_bench


def test_effective_slot_context_uses_llama_cpp_per_slot_budget():
    runtime_summary = {
        "backend_format": "llama_cpp_gguf",
        "loaded_context_length": 32768,
        "ctx_size": 32768,
        "parallel_slots": 2,
    }

    effective = concurrency_bench._effective_slot_context_tokens(runtime_summary, 65536)

    assert effective == 16384


def test_effective_slot_context_preserves_single_slot_context():
    runtime_summary = {
        "backend_format": "llama_cpp_gguf",
        "loaded_context_length": 32768,
        "ctx_size": 32768,
        "parallel_slots": 1,
    }

    effective = concurrency_bench._effective_slot_context_tokens(runtime_summary, 65536)

    assert effective == 32768
