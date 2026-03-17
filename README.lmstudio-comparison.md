# DGX Spark Comparison: `omlx_dgx + llama.cpp` vs `LM Studio`

This document tracks the latest apples-to-apples comparison between the DGX
Spark path implemented in this repo and the local `LM Studio` baseline.

## Scope

This comparison is intentionally narrow:

- hardware: DGX Spark / NVIDIA GB10
- context window: `32768`
- quantization: `Q4_K_M`
- model size: `4B`
- backend family: GGUF

The current local machine did not have an exact `Qwen3 4B` text model loaded
in `LM Studio`, so this comparison uses the closest fair baseline that was
actually available on both sides:

- `omlx_dgx`: `Qwen3.5-4B-Q4_K_M.gguf`
- `LM Studio`: `qwen3.5-4b@q4_k_m`

That means the comparison is fair on model file, size, quantization, and
context length. It is not a comparison between different Qwen generations.

## Compared Configurations

### `omlx_dgx`

Single-session benchmark:

- backend: `llama.cpp`
- preset: `single_session_low_latency`
- `parallel_slots=1`
- `ctx_size=32768`
- `batch_size=8192`
- `ubatch_size=2048`
- `checkpoint_every_n_tokens=1024`
- `ctx_checkpoints=64`
- `cache_ram_mib=16384`

Mixed-traffic benchmark:

- backend: `llama.cpp`
- preset: `mixed_traffic`
- `parallel_slots=2`
- `ctx_size=32768`

### `LM Studio`

- loaded model: `qwen3.5-4b@q4_k_m`
- `loaded_context_length=32768`
- benchmarked through the local OpenAI-compatible API at `http://127.0.0.1:1234`

## Workloads

The following cases were benchmarked:

- `short_chat`
- `long_output_chat`
- `long_prefix_run_1`
- `long_prefix_run_2`
- `single_session_followup`
- `dual_short_independent`
- `dual_long_output_independent`
- `dual_long_prefix_independent`
- `repeat_long_prefix_plus_short`

## Results

### Single Session

| Metric | `omlx_dgx + llama.cpp` | `LM Studio` |
| --- | ---: | ---: |
| `short_chat` | `0.374s` | `0.541s` |
| `long_output_chat` | `24.041 tok/s` | `27.129 tok/s` |
| `long_prefix_run_1` | `4.528s` | `3.186s` |
| `long_prefix_run_2` | `0.227s` | `0.244s` |
| `long_prefix_speedup` | `19.95x` | `13.06x` |
| `multi_turn_chat` | `18.757 tok/s` | `29.442 tok/s` |
| `followup_avg` | `0.146s` | `0.436s` |
| `turn2_short_followup` | `0.168s` | `0.368s` |
| `turn3_short_followup` | `0.124s` | `0.504s` |
| `turn2_prompt_tokens` | `50` | `4865` |
| `turn3_prompt_tokens` | `40` | `4888` |

### Mixed Traffic

| Metric | `omlx_dgx + llama.cpp` | `LM Studio` |
| --- | ---: | ---: |
| `dual_short_makespan` | `0.185s` | `1.646s` |
| `dual_long_output_makespan` | `7.183s` | `13.521s` |
| `dual_long_output_tps` | `53.457 tok/s` | `28.401 tok/s` |
| `dual_long_prefix_makespan` | `2.903s` | `2.855s` |
| `repeat_plus_short_makespan` | `0.318s` | `1.281s` |
| `repeat_speedup_x` | `12.24x` | `2.31x` |
| `repeat_long_sec` | `0.224s` | `1.279s` |
| `concurrent_short_sec` | `0.316s` | `0.945s` |
| `warmup_long_sec` | `2.741s` | `2.952s` |

## Interpretation

### Where `omlx_dgx` is currently better

Single-session continued chat:

- much lower follow-up latency after a long context turn
- stronger repeated long-prefix reuse
- lower prompt token count on later turns because continuation compression is
  active

Mixed traffic:

- much better `repeat_long_prefix_plus_short` makespan
- much faster repeated long request under concurrent pressure
- much faster concurrent short request handling
- stronger repeated-prefix speedup when another request arrives

The current advantage comes mainly from the repo-specific logic layered on top
of `llama.cpp`:

- continuation routing
- anchored prompt compression
- slot stickiness
- mixed-traffic slot protection

### Where `LM Studio` is still better

- colder long-prefix first turn latency
- pure long-output throughput in the single-session benchmark
- multi-turn throughput when the full history is still being resent each turn

This means `LM Studio` is still stronger when the main objective is:

- first-turn responsiveness
- raw decode throughput
- simple local chat without aggressive continuation-aware optimization

## Practical Conclusion

If the target workload is:

- long-context coding chat
- repeated follow-up turns
- avoiding full recomputation of old context
- long-running chat mixed with short side requests

then the current `omlx_dgx + llama.cpp` path is already ahead of the local
`LM Studio` baseline on the same `Q4_K_M` model artifact.

If the target workload is:

- cold-start speed
- pure long-form generation throughput

then `LM Studio` still has an advantage.

## Raw Benchmark Artifacts

The latest raw outputs used for this comparison were generated from:

- `/tmp/omlx_vs_lmstudio_omlx_q4km.json`
- `/tmp/omlx_vs_lmstudio_lmstudio_q4km.json`
- `/tmp/omlx_vs_lmstudio_omlx_q4km_mixed.json`
- `/tmp/omlx_vs_lmstudio_lmstudio_q4km_concurrency.json`
