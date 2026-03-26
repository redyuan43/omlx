# Qwen3.5-35B on Jetson AGX Orin: oMLX vs Ollama (2026-03-26 Rerun)

This document records the isolated 35B rerun on `2026-03-26`. `oMLX` and
`Ollama` were not left running at the same time: the `oMLX` phase completed
first, all `8008/8018/30000/30100` listeners were stopped, and only then was
`ollama` restarted for the comparison phase.

## Scope

- Hardware: Jetson AGX Orin 64GB
- Model class: `Qwen3.5-35B`
- Quantization: `Q4_K_M`
- Comparison scope: text-only LLM benchmarks
- `oMLX` single-slot profile: `32K`
- `oMLX` dual-slot mixed profile: effective `16K` per slot
- Ollama tag used in this rerun: `qwen3.5:35b`

Artifacts:

- `oMLX think off`: `/tmp/bench_qwen35_35b_omlx_ctx32768_think_off.json`
- `oMLX think on`: `/tmp/bench_qwen35_35b_omlx_ctx32768_think_on.json`
- `oMLX dual-slot`: `/tmp/bench_qwen35_35b_concurrency_parallel2.json`
- `Ollama think off`: `/tmp/bench_qwen35_35b_ollama_think_off.txt`
- `Ollama think on`: `/tmp/bench_qwen35_35b_ollama_think_on.txt`
- `Ollama dual-request 32K`: `/tmp/bench_qwen35_35b_ollama_concurrency.json`
- `Ollama dual-request 16K`: `/tmp/bench_qwen35_35b_ollama_concurrency_ctx16384.json`

## Single-Flow 32K

The warm-path comparison is the useful one here. `oMLX think off short_chat`
included a cold model load (`28.207s`) and is not directly comparable.

| Metric | oMLX | Ollama |
| --- | ---: | ---: |
| `short_chat` warm path | `0.431s` | `1.713s` |
| `long_output_chat` | `7.445s` | `9.071s` |
| `long_prefix_run_1` | `35.903s` | `103.904s` |
| `long_prefix_run_2` | `0.299s` | `103.811s` |
| turn 2 follow-up | `0.317s` | `2.272s` |
| turn 3 follow-up | `0.306s` | `0.737s` |
| post-restart turn 4 | `0.358s` | `0.870s` |

Takeaway: on the same 35B class, `oMLX` is clearly better at long-prefix reuse
and warm follow-up latency.

## Dual-Parallel Fair Comparison

The `oMLX` mixed profile uses `parallel_slots=2`, but the runtime effectively
exposes `16384` tokens per slot. The fair comparison is therefore:

- `oMLX 2-slot @ 16K/slot`
- `Ollama 2 concurrent requests @ 16K`

| Metric | oMLX | Ollama |
| --- | ---: | ---: |
| dual short makespan | `0.740s` | `0.957s` |
| dual long output makespan | `14.928s` | `17.773s` |
| dual long prefix makespan | `14.585s` | `62.839s` |
| warmup long prefix | `13.517s` | `52.291s` |
| repeated long prefix | `1.063s` | `31.934s` |
| concurrent short beside repeat | `0.313s` | `0.734s` |
| repeat speedup | `12.72x` | `1.64x` |

Takeaway: `oMLX` is materially stronger on mixed long-prefix traffic. Ollama
showed some benefit at `16K`, but still behaved much closer to serialized work
than to true cache reuse.

## Restart Recovery Note

This rerun exposed an important runtime limitation. The `oMLX` benchmark
recorded:

- `turn3_restore_status = restore_error`
- backend error: `501 ... This feature is not supported by multimodal`

That error does not mean the model cannot do multimodal inference. It means the
current `llama.cpp` slot `save/restore` API is not available for this multimodal
model family. The impact is narrow:

- normal chat still works
- in-process continuation still works
- restart-time slot restoration does not

The codebase now auto-detects this backend response and disables session
restore, cold-cache restore, and named-context restore attempts for the running
adapter instead of repeatedly surfacing `restore_error`.

## Bottom Line

- For warm 35B text chat, `oMLX` remains faster than `Ollama`.
- For long-prefix reuse and dual mixed traffic, `oMLX` is decisively better.
- For restart recovery on this multimodal 35B runtime, slot restoration must be
  treated as unavailable, so replay remains the correct fallback path.
