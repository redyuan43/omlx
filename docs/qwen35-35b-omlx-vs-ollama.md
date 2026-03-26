# Qwen3.5-35B on Jetson AGX Orin: oMLX vs Ollama

> Status: this file is retained as an earlier comparison snapshot. For the
> latest isolated rerun and updated 35B conclusions, see
> [qwen35-35b-omlx-vs-ollama-20260326.md](/home/agx/github/omlx/docs/qwen35-35b-omlx-vs-ollama-20260326.md).
> The `2026-03-26` rerun includes isolated `oMLX` and `Ollama` phases, a fair
> `16K` dual-parallel comparison, and the current note about multimodal
> `llama.cpp` slot `save/restore` support.

This document compares the current `oMLX + llama.cpp` path against the
current `Ollama` path for `Qwen3.5-35B` on the same Jetson AGX Orin machine.

## Test Scope

- Hardware: Jetson AGX Orin 64GB
- Model class: `Qwen3.5-35B`
- Quantization: `Q4_K_M`
- Primary `oMLX` runtime path: `llama.cpp`
- Primary `oMLX` stable context: `32768`
- Comparison scope: text inference only

Artifacts used for the comparison:

- `oMLX think off`: `/tmp/bench_qwen35_35b_omlx_ctx32768_think_off.json`
- `oMLX think on`: `/tmp/bench_qwen35_35b_omlx_ctx32768_think_on.json`
- `Ollama think off`: `/tmp/bench_qwen35_35b_ollama_think_off.txt`
- `Ollama think on`: `/tmp/bench_qwen35_35b_ollama_think_on.txt`

## Runtime Setup

| Item | oMLX | Ollama |
| --- | --- | --- |
| Runtime path | `oMLX + llama.cpp` | `Ollama` |
| Model | `Qwen3.5-35B-A3B Q4_K_M GGUF` | `qwen3.5:35b Q4_K_M` |
| Stable tested context | `32768` | not exposed in the summary artifact |
| Long-prefix continuation reuse | yes | not evident in this run |
| Session restore | yes | limited restart path only |
| Incremental `+78 token` case | yes | not present in current 35B artifact |
| RAG integration | yes, current stack | not part of this benchmark |
| PDF/OCR integration | yes, current stack | not part of this benchmark |

## Think Off

| Metric | oMLX | Ollama | Delta |
| --- | ---: | ---: | --- |
| `short_chat` | `0.350s` | `0.501s` | oMLX `1.43x` faster |
| `long_output_chat` | `7.320s` | `8.176s` | oMLX `1.12x` faster |
| `long_output decode` | `26.23 tok/s` | `25.48 tok/s` | oMLX slightly faster |
| `long_prefix_run_1` | `36.928s` | `102.883s` | oMLX `2.79x` faster |
| `long_prefix_run_2` | `1.289s` | `102.898s` | oMLX `79.8x` faster |
| long-prefix repeat speedup | `28.65x` | about `1.00x` | oMLX much better |
| warm follow-up avg | `0.264s` | `1.404s` | oMLX `5.32x` faster |
| restart time | `0.963s` | `2.167s` | oMLX `2.25x` faster |
| ready after restart | `1.001s` | `1.003s` | effectively equal |
| post-restart replay | `0.422s` | `16.230s` | oMLX `38.5x` faster |
| post-restore follow-up | `0.255s` | `0.665s` | oMLX `2.61x` faster |

## Think On

| Metric | oMLX | Ollama | Delta |
| --- | ---: | ---: | --- |
| `short_chat` | `0.754s` | `1.088s` | oMLX `1.44x` faster |
| `long_output_chat` | `7.230s` | `8.208s` | oMLX `1.14x` faster |
| `long_output decode` | `26.56 tok/s` | `25.36 tok/s` | oMLX slightly faster |
| `long_prefix_run_1` | `37.260s` | `104.091s` | oMLX `2.79x` faster |
| `long_prefix_run_2` | `1.349s` | `104.309s` | oMLX `77.3x` faster |
| long-prefix repeat speedup | `27.62x` | about `1.00x` | oMLX much better |
| warm follow-up avg | `0.256s` | `1.415s` | oMLX `5.53x` faster |
| restart time | `0.875s` | `1.900s` | oMLX `2.17x` faster |
| ready after restart | `1.002s` | `1.003s` | effectively equal |
| post-restart replay | `0.403s` | `14.983s` | oMLX `37.2x` faster |
| post-restore follow-up | `0.252s` | `0.662s` | oMLX `2.63x` faster |

## Incremental Long-Context

The current `oMLX` benchmark includes a stricter incremental case: the
second request reuses the same long base prompt and appends exactly `78`
tokens. The current `Ollama` 35B artifact does not include an equivalent
case.

| Metric | oMLX Think Off | oMLX Think On | Ollama |
| --- | ---: | ---: | --- |
| base long prompt run | `29.611s` | `30.024s` | not measured in this artifact |
| same base `+78 token` | `1.419s` | `1.406s` | not measured |
| speedup | `20.87x` | `21.36x` | not measured |
| actual delta tokens | `78` | `78` | not measured |

## Thinking Overhead

| Runtime | short request overhead | long output impact | follow-up impact |
| --- | --- | --- | --- |
| oMLX | `0.350s -> 0.754s` | negligible | negligible |
| Ollama | `0.501s -> 1.088s` | negligible | negligible |

## Summary

| Area | Result |
| --- | --- |
| Raw 35B usability | both run |
| Text latency | `oMLX` better |
| Long-context reuse | `oMLX` much better |
| Warm chat experience | `oMLX` much better |
| Restart recovery | `oMLX` much better |
| Incremental long-context reuse | validated on `oMLX` |
| RAG / PDF / OCR integration | currently on the `oMLX` side |
| Simplicity | `Ollama` better |

## Bottom Line

For `Qwen3.5-35B` on this Jetson, the current `oMLX` path is decisively
better than `Ollama` on the metrics that matter for long-context reuse,
warm follow-up latency, and restart recovery.
