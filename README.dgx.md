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
- `Q4_K_S`
- `parallel_slots=1` for single-session use
- `parallel_slots=2 + session stickiness` for mixed traffic

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

Measured on `Qwen3.5-4B-Q4_K_S.gguf`:

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

- Keep `parallel_slots=1` as the default preset for single-session local chat.
- Keep `parallel_slots=2 + session stickiness` as an experimental preset for mixed traffic.
- Benchmark `Q4_K_M` and `Q6_K` under the same mixed-traffic workload before changing the default GGUF variant.
