# Qwen3.5-35B Jetson Stable Checkpoint

This document records the current stable checkpoint for the `Qwen3.5-35B` `oMLX + llama.cpp` path on Jetson AGX Orin.

## Stable Default

- backend: `oMLX + llama.cpp`
- model: `Qwen3.5-35B-A3B.Q4_K_M.gguf`
- context: `32768`
- `parallel_slots=1`
- keep:
  - single-slot long-prefix reuse
  - restart recovery
  - cold-cache lookup with narrowed candidate selection
- do not promote:
  - `parallel_slots=2` mixed-traffic mode

## Why `parallel_slots=2` Is Not the Default

Real mixed-traffic testing showed that `parallel_slots=2` can run, but it effectively split the runtime into two `16384`-token slots instead of preserving a full `32768`-token slot.

That is acceptable as an experiment, but not as the stable production path for the current 35B target.

Artifact:
- [/tmp/bench_qwen35_35b_concurrency_parallel2.json](/tmp/bench_qwen35_35b_concurrency_parallel2.json)

Key observations:
- dual long-prefix makespan: `13.703s`
- repeat long-prefix plus concurrent short request:
  - warmup: `13.229s`
  - repeat: `0.939s`
  - short: `0.281s`
  - speedup: `14.09x`
- runtime slot summary showed `n_ctx=16384` per slot

Conclusion:
- useful experimental signal
- not acceptable as the stable `35B 32K` default

## Stable Benchmark Snapshot

Baseline artifact:
- [/tmp/bench_qwen35_35b_omlx_ctx32768_think_off.json](/tmp/bench_qwen35_35b_omlx_ctx32768_think_off.json)

Current stable checkpoint artifact:
- [/tmp/bench_qwen35_35b_phase12_opt2.json](/tmp/bench_qwen35_35b_phase12_opt2.json)

### Core Metrics

| Metric | Baseline | Stable Checkpoint |
| --- | ---: | ---: |
| `short_chat` | `0.350s` | `0.280s` |
| `long_prefix_run_1` | `36.928s` | `36.744s` |
| `long_prefix_run_2` | `1.289s` | `1.328s` |
| `+78 token incremental` | `1.419s` | `1.417s` |
| `warm follow-up avg` | `0.264s` | `0.262s` |
| `post-restart replay` | `0.422s` | `0.390s` |
| `post-restore follow-up` | `0.255s` | `0.255s` |

## What Was Preserved

- cold-path regression from the earlier `phase12` attempt was removed
- hot-path behavior remained near the original stable baseline
- recovery remained correct
- no new correctness regressions were introduced

## What Was Rejected

- broad `parallel_slots=2` mixed-traffic promotion
- any optimization that improved hot reuse while materially regressing `long_prefix_run_1`

## Current Engineering Decision

Keep the current checkpoint as the working stable base for future 35B work:

1. single-slot `32K`
2. careful shared-prefix improvements only if they do not regress cold-path latency
3. no mixed-traffic promotion until a future design preserves the full effective context budget

## Validation

Broader regression suite run on this checkpoint:

- `tests/test_omlx_dgx_session_restore.py`
- `tests/test_omlx_dgx_llama_cpp_adapter.py`
- `tests/test_omlx_dgx_app.py`

Result:
- `55 passed`
