# oMLX DGX Experiment

This repository now contains an experimental `omlx_dgx` package for the DGX
runtime path:

- SGLang-oriented control-plane scaffolding for `Qwen/Qwen3.5-35B-A3B`
- Managed SGLang runtime adapter with HiCache status/attach/detach support
- Experimental TensorRT-LLM adapter kept as a research backend
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

## Package Layout

- `omlx_dgx/cache_core.py`: block hashes, LRU queue, content-addressed block ledger
- `omlx_dgx/tiered_kv.py`: GPU/host/SSD tier metadata and persistent cold-store manifests
- `omlx_dgx/scheduler_policy.py`: prefix-aware admission/sorting/eviction policy
- `omlx_dgx/runtime/backend.py`: backend adapter contract and HTTP adapter
- `omlx_dgx/runtime/sglang.py`: SGLang adapter, HiCache admin surface, start/stop/logs
- `omlx_dgx/runtime/tensorrt_llm.py`: experimental TensorRT-LLM adapter, diagnostics, start/stop/logs
- `omlx_dgx/control_plane/app.py`: FastAPI control-plane and admin endpoints
- `omlx_dgx/cli.py`: `omlx-dgx` launcher

## Next Runtime Hook Points

- Validate the SGLang runtime venv and `sgl-kernel` wheel path on this DGX Spark host.
- Feed real HiCache/L3 events into a backend-specific cold-cache manifest bridge.
- Wire scheduler decisions from `OmlxSchedulerPolicy` into a deeper runtime hook when kernel changes are justified.
