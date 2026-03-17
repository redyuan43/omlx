# Current DGX Spark Baseline

Mainline:

- `Qwen3.5-4B GGUF + llama.cpp + Q4_K_M`
- default llama.cpp preset: `single_session_low_latency`
- concurrent preset: `mixed_traffic`
- current recommended `ctx_size`: `32768`

Measured single-session numbers for `Q4_K_M` on DGX Spark:

- `short_chat`: `0.161s`
- `long_output`: `53.333 tok/s`
- `long_prefix_run_1`: `2.082s`
- `long_prefix_run_2`: `0.066s`
- `turn2_short_followup`: `0.124s`
- `turn3_short_followup`: `0.099s`
- `single_session_followup_avg`: `0.111s`

Measured mixed-traffic numbers for `Q4_K_M`:

- `repeat_long_prefix + short` makespan: `0.158s`
- `repeat_speedup_x`: `22.877x`
- repeated long request: `0.125s`
- concurrent short request: `0.115s`

LM Studio comparison on `Q4_K_M` and `32k` context:

- `short_chat`: `0.409s`
- `long_output`: `59.390 tok/s`
- `long_prefix_run_1`: `1.864s`
- `long_prefix_run_2`: `0.135s`
- `turn2_short_followup`: `0.225s`
- `turn3_short_followup`: `0.251s`
- `single_session_followup_avg`: `0.238s`

Important runtime facts:

- `Qwen3.5-4B` currently follows a hybrid/recurrent path in `llama.cpp`.
- `cache_reuse` is reported as unsupported for this context.
- Current wins come mainly from repo-side logic:
  - single-session continuation
  - anchored prompt compression
  - slot-aware routing
  - mixed-traffic stickiness and recycling

Presets already present in the repo:

- `single_session_low_latency`
  - `parallel_slots=1`
  - continuation enabled
- `mixed_traffic`
  - `parallel_slots=2`
  - session stickiness enabled
