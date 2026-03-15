# oMLX DGX Experiment

This repository now contains an experimental `omlx_dgx` package for the DGX
runtime path:

- SGLang-oriented control-plane scaffolding for `Qwen/Qwen3.5-35B-A3B`
- Managed SGLang runtime adapter with HiCache status/attach/detach support
- Experimental TensorRT-LLM adapter kept as a research backend
- Managed `llama.cpp` GGUF runtime for `Qwen3.5-4B`
- Backend-agnostic block metadata and scheduler policy ported from oMLX ideas
- Tiered KV manifest store with cross-restart SSD metadata persistence
- OpenAI-compatible HTTP proxy endpoints for `/v1/chat/completions` and `/v1/completions`
- A minimal web console at `/admin`

## Quick Start

```bash
omlx-dgx serve \
  --backend-kind sglang \
  --backend-url http://127.0.0.1:30000 \
  --runtime-python /path/to/sglang-venv/bin/python \
  --model-repo-id Qwen/Qwen3.5-35B-A3B \
  --model-id qwen35-35b \
  --model-alias qwen35
```

This does not yet fork SGLang or TensorRT-LLM internals. Instead, it provides
the control plane and runtime/storage interfaces that the DGX path can build on
next.

## Current Recommendation

For `Qwen3.5-4B` quantized serving on DGX Spark, the current best path in this
repo is:

- `GGUF + llama.cpp`
- `Q4_K_M`
- `single_session_low_latency` preset for single-session use
- `mixed_traffic` preset for long-context + short-request concurrency
- `ctx_size=32768` as the current DGX Spark default for llama.cpp presets

This keeps the original oMLX idea at the control-plane layer: do not let short
requests blow away a long conversation's warm prefix.

## Measured Advantages

### Single Session

For a single long-context chat, `parallel_slots=1` remains the safest default.
It keeps cold long-prefix latency lower while still preserving strong reuse on
repeat turns:

- long prefix first run: `1.566s`
- same long prefix second run: `0.070s`
- repeat speedup: `22.37x`

### Mixed Traffic

The more interesting result is mixed traffic: one long-context conversation is
already warm, then a short request arrives at the same time.

With `parallel_slots=2 + session stickiness`, the control plane now pins the
long session to its slot and routes the short request away from that hot prefix.
After fixing stale-slot recycling, the mixed benchmark became stable.

Measured on `Qwen3.5-4B` GGUF variants:

- `Q4_K_M`
  - single-session follow-up avg: `0.111s`
  - long output throughput: `53.333 tok/s`
  - long prefix run 1 / run 2: `2.082s / 0.066s`
- `Q4_K_S`
  - single-session follow-up avg: `0.115s`
  - long output throughput: `54.432 tok/s`
  - long prefix run 1 / run 2: `2.075s / 0.067s`
- `Q6_K`
  - single-session follow-up avg: `0.135s`
  - long output throughput: `45.077 tok/s`
  - long prefix run 1 / run 2: `2.163s / 0.075s`

For the current DGX Spark workload, `Q4_K_M` is the recommended default because
it gives the best long-context follow-up latency with only a minor throughput
tradeoff versus `Q4_K_S`.

Compared against `LM Studio` on the same `Q4_K_M` quantization and `32k`
context, the current single-session follow-up path in `omlx_dgx + llama.cpp`
is faster on continued turns:

- `omlx_dgx + llama.cpp`
  - turn2 follow-up: `0.124s`
  - turn3 follow-up: `0.099s`
  - follow-up avg: `0.111s`
- `LM Studio`
  - turn2 follow-up: `0.225s`
  - turn3 follow-up: `0.251s`
  - follow-up avg: `0.238s`
- `omlx_dgx + llama.cpp`
  - warm long request: `1.764s`
  - repeated long request: `0.097s`
  - concurrent short request: `0.176s`
  - total mixed makespan: `0.192s`
  - repeat speedup: `21.3x`
- `LM Studio`
  - warm long request: `1.419s`
  - repeated long request: `0.131s`
  - concurrent short request: `0.366s`
  - total mixed makespan: `0.367s`
  - repeat speedup: `11.58x`

In that mixed scenario, the current `omlx_dgx + llama.cpp` stack is faster than
LM Studio:

- repeated long request: about `1.35x`
- concurrent short request: about `2.08x`
- total mixed makespan: about `1.91x`

### What Improved

The recent `llama.cpp` work did not replace llama.cpp's cache engine. Instead,
it added oMLX-style scheduling on top:

- conversation stickiness for long prompts
- explicit slot assignment for warm follow-up requests
- stale idle-slot recycling for unrelated short requests
- benchmark and admin diagnostics for slot routing decisions

That means the current advantage is strongest when your workload includes long
conversations plus unrelated short requests or multiple chats at once.

## DGX Spark Presets

The managed `llama.cpp` adapter now exposes two named presets:

- `single_session_low_latency`
  - `parallel_slots=1`
  - `ctx_size=32768`
  - single-session continuation enabled
  - default DGX Spark preset for local chat
- `mixed_traffic`
  - `parallel_slots=2`
  - `ctx_size=32768`
  - slot stickiness enabled
  - intended for long-context plus short-request concurrency

You can select them from the CLI:

```bash
omlx-dgx serve \
  --backend-kind llama_cpp \
  --quant-mode gguf_experimental \
  --model-source gguf \
  --artifact-path /models/Qwen3.5-4B-Q4_K_M.gguf \
  --serving-preset single_session_low_latency
```

## Telemetry

`/admin/api/runtime` now preserves the low-level GPU telemetry state instead of
silently hiding it. On this DGX Spark, if `nvidia-smi` fails with an NVML error,
you will see it under:

- `backend.details.telemetry.gpu_metrics_ok`
- `backend.details.telemetry.gpu_metrics_error`
- `backend.details.telemetry.system_memory_kb`

This makes it possible to distinguish "no GPU info because nothing is wrong"
from "no GPU info because NVML is currently broken".

## Matrix Benchmarking

Use the matrix script to compare `Q4_K_S`, `Q4_K_M`, `Q6_K` and `16k/32k`
variants across both single-session follow-up and mixed traffic:

```bash
python3 scripts/bench_qwen35_4b_matrix.py \
  --omlx-variant q4ks-16k,http://127.0.0.1:8020,http://127.0.0.1:31200 \
  --omlx-variant q4km-32k,http://127.0.0.1:8021,http://127.0.0.1:31201 \
  --omlx-variant q6k-32k,http://127.0.0.1:8022,http://127.0.0.1:31202 \
  --lmstudio-variant lmstudio,http://127.0.0.1:1234,qwen3.5-4b \
  --include-concurrency
```

The output now includes:

- `serving_preset`
- telemetry health/error fields
- single-session follow-up latency
- mixed-traffic makespan
- automatic recommendations for:
  - best single-session follow-up
  - best mixed-traffic profile
  - best long-output throughput
  - best cold long-prefix latency

## Package Layout

- `omlx_dgx/cache_core.py`: block hashes, LRU queue, content-addressed block ledger
- `omlx_dgx/tiered_kv.py`: GPU/host/SSD tier metadata and persistent cold-store manifests
- `omlx_dgx/scheduler_policy.py`: prefix-aware admission/sorting/eviction policy
- `omlx_dgx/runtime/backend.py`: backend adapter contract and HTTP adapter
- `omlx_dgx/runtime/sglang.py`: SGLang adapter, HiCache admin surface, start/stop/logs
- `omlx_dgx/runtime/tensorrt_llm.py`: experimental TensorRT-LLM adapter, diagnostics, start/stop/logs
- `omlx_dgx/runtime/llama_cpp.py`: llama.cpp GGUF adapter with slot-aware routing and process management
- `omlx_dgx/control_plane/app.py`: FastAPI control-plane and admin endpoints
- `omlx_dgx/cli.py`: `omlx-dgx` launcher

## Next Runtime Hook Points

- Keep `single_session_low_latency` as the default preset for local DGX Spark chat.
- Keep `mixed_traffic` as the experimental preset for concurrent workloads.
- Keep `Q4_K_M` as the default GGUF variant unless your workload is dominated by long-output throughput.
