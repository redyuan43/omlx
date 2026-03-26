# oMLX DGX Experiment

This repository now contains an experimental `omlx_dgx` package for the DGX
runtime path:

- SGLang-oriented control-plane scaffolding for `Qwen/Qwen3.5-35B-A3B`
- Managed SGLang runtime adapter with HiCache status/attach/detach support
- Experimental TensorRT-LLM adapter kept as a research backend
- Managed `llama.cpp` GGUF runtime for `Qwen3.5-4B`
- Managed `llama.cpp` GGUF model pool with pin/TTL/idle unload/LRU/manual load-unload control
- Managed `llama.cpp` session restore via slot save/restore plus persisted cold metadata snapshots
- Backend-agnostic block metadata and scheduler policy ported from oMLX ideas
- Tiered KV manifest store with cross-restart SSD metadata persistence
- Capability-aware HTTP service endpoints for `/v1/chat/completions`,
  `/v1/completions`, `/v1/messages`, `/v1/embeddings`, and `/v1/rerank`
  with model-level gating per registered backend
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

For the current local Jetson `35B + RAG + OCR` usage guide, see
[docs/jetson-local-service-guide.md](docs/jetson-local-service-guide.md).

## Mac Path vs DGX Path

The DGX path is not a full port of the Mac runtime. Today it reuses oMLX
control-plane ideas on top of DGX backends, rather than reusing the Apple
Silicon MLX/Metal inference stack directly.

### 1. Mac/oMLX Optimizations

The Mac path in [README.md](README.md) includes:

- MLX-native inference via `MLX`, `mlx-lm`, and `mlx-vlm`
- Apple Silicon and Metal specific runtime/kernel optimizations
- the native oMLX runtime path for continuous batching
- hot-memory plus cold-SSD KV caching with prefix sharing and paged cache style behavior
- the full Mac product layer: admin UI, model downloader, integrations, and menu bar app

### 2. What DGX/Jetson Already Reuses

The DGX path already carries over these oMLX ideas:

- control-plane scheduling and runtime orchestration
- conversation stickiness for long prompts
- slot assignment for warm follow-up requests
- slot save/restore plus persisted session metadata
- managed model-pool behavior: pin, TTL, idle unload, LRU, and manual load/unload
- capability-aware routing for chat, completions, messages, embeddings, rerank, VLM, and OCR
- backend-agnostic block metadata, scheduler policy, and tiered-KV manifest metadata

### 3. What Is Not Ported Yet

These parts of the Mac path are still Apple-only or otherwise absent on DGX:

- the MLX runtime itself
- Apple Silicon and Metal execution kernels
- the native Mac cache engine implementation
- a backend replacement for `llama.cpp` cache internals
- the macOS app and menu bar integration layer

For the current `llama.cpp` path, the important detail is that DGX adds
oMLX-style scheduling on top of `llama.cpp`; it does not replace
`llama.cpp`'s cache engine.

### 4. Highest-Value Next Steps For Jetson

For the current Jetson AGX Orin workload, the most valuable next ports are:

- a real cache-reuse or prefix-cache path that works with `Qwen3.5` hybrid recurrent models
- stronger incremental long-context reuse for "same base prompt plus small delta" requests
- a more concrete tiered-KV cold-store path beyond metadata and restore manifests
- broader concurrent scheduling once the single-session and long-prefix path is stable

## Current DGX Capability Matrix

The current DGX path now supports four capability classes behind the same
control-plane:

- text chat/completions:
  - primary path: `llama.cpp + Qwen3.5-4B GGUF`
- embeddings:
  - current live path: external OpenAI-compatible embedding model registration
  - validated against local `LM Studio` with
    `text-embedding-nomic-embed-text-v1.5`
- VLM and OCR:
  - current live path: external OpenAI-compatible model registration
  - validated against local `LM Studio` with `qwen/qwen3-vl-8b` and
    `glm-ocr@q4_k_s`
- rerank:
  - current live path: managed `llama.cpp --reranking`
  - validated against `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`

On DGX, these capabilities are model-scoped rather than backend-global. A model
registered as `embeddings` will not be allowed to answer chat requests, and a
reranker will not be exposed as a chat model.

## Current Recommendation

For `Qwen3.5-4B` quantized serving on DGX Spark, the current best path in this
repo is:

- `GGUF + llama.cpp`
- `Q4_K_M`
- `single_session_low_latency` preset for single-session use
- `mixed_traffic` preset for long-context + short-request concurrency
- `ctx_size=32768` as the current DGX Spark default for llama.cpp presets

For deeper single-session benchmarking, keep the preset defaults above but
launch an explicit `ctx_size=65536` service. The current benchmark scripts now
auto-target the runtime's effective context window and leave enough headroom
for follow-up turns and short-restart recovery.

Phase 4 also makes benchmark reports available through stable admin APIs:

- `GET /admin/api/runtime/capabilities`
- `GET /admin/api/benchmarks`
- `POST /admin/api/benchmarks/qwen35-4b/run`
- `GET /admin/api/benchmarks/qwen35-4b/latest`
- `GET /admin/api/benchmarks/qwen35-4b/reports`

This keeps the original oMLX idea at the control-plane layer: do not let short
requests blow away a long conversation's warm prefix.

Phase 2 also adds a DGX-side GGUF model pool on top of the managed
`llama.cpp` path:

- multi-model registration persisted under `settings.json`
- per-model pin, TTL, and idle unload policy
- LRU eviction when the pool is at capacity
- manual load/unload/pin admin actions
- `/admin/api/runtime/model-pool` diagnostics showing loaded state, last use,
  pin state, TTL/idle policy, and unload/eviction reasons

## Measured Advantages

### Single Session

For a single long-context chat, `parallel_slots=1` remains the safest default.
It keeps cold long-prefix latency lower while still preserving strong reuse on
repeat turns:

- long prefix first run: `1.566s`
- same long prefix second run: `0.070s`
- repeat speedup: `22.37x`

### 64k Extension And Recovery

On the current Phase 3 `Q4_K_M` path, a managed single-session service with
`ctx_size=65536` now persists a cold metadata snapshot for the warm chat and
restores the saved llama.cpp slot after a short managed restart:

- benchmark target context: `65536`
- runtime-reported cold long prompt: `56200` prompt tokens
- cold long-prefix run: `40.334s`
- repeated long-prefix run: `1.134s`
- repeat speedup: `35.57x`
- warm follow-up before restart: `0.307s`
- post-restart restored follow-up: `3.285s`
- slot restore RPC: `7.101ms`
- post-restore next follow-up: `0.185s`
- cold-to-restored speedup: `12.03x`

This is not GGUF block-level KV restore. It is managed slot save/restore plus
persisted continuation metadata, with safe fallback back to cold replay if the
saved slot cannot be restored.

For the latest same-artifact comparison against the local `LM Studio` baseline,
see [README.lmstudio-comparison.md](README.lmstudio-comparison.md).

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

On the current Phase 3 managed `mixed_traffic` run, the control plane still
keeps the short request off the warm long slot:

- configured `ctx_size=65536`, runtime-reported per-slot context: `32768`
- warm long request: `13.123s`
- repeated long request: `0.548s`
- concurrent short request: `0.666s`
- total mixed makespan: `0.668s`
- repeat speedup: `23.95x`

This is why the concurrency benchmark now keys off the runtime's effective
per-slot context instead of assuming the configured `ctx_size` is available to
every slot.

### Multi-Model Pool

The new model-pool layer keeps the primary long-context chat warm by giving
each registered GGUF model its own managed `llama-server` process and policy
state instead of swapping one global runtime in place.

On the current DGX Spark Phase 2 validation run:

- primary `Q4_K_M` long prompt: `12018` estimated prompt tokens
- primary cold long turn: `9.336s`
- primary warm follow-up before secondary activity: `0.402s`
- secondary `Q4_K_S` manual load: `2.016s`
- secondary short request: `0.242s`
- secondary manual unload: `0.009s`
- primary warm follow-up after secondary load/unload: `0.279s`
- admin diagnostics after the run:
  - primary stayed `loaded=true`, `pinned=true`
  - secondary ended `loaded=false`, `last_unload_reason=manual_unload`
  - continuation state reported `last_decision.reason=hit`

### What Improved

The recent `llama.cpp` work did not replace llama.cpp's cache engine. Instead,
it added oMLX-style scheduling on top:

- conversation stickiness for long prompts
- explicit slot assignment for warm follow-up requests
- managed slot save/restore after short runtime restarts
- stale idle-slot recycling for unrelated short requests
- benchmark and admin diagnostics for slot routing and restore decisions

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

Additional GGUF models are registered from the admin API for now. Example:

```bash
curl -X POST http://127.0.0.1:8008/admin/api/runtime/model-pool \
  -H 'Content-Type: application/json' \
  -d '{
    "model_id": "secondary",
    "model_alias": "qwen35-secondary",
    "artifact_path": "/models/Qwen3.5-4B-Q4_K_S.gguf",
    "base_url": "http://127.0.0.1:30001",
    "gguf_variant": "Q4_K_S",
    "ctx_size": 16384,
    "parallel_slots": 1,
    "ttl_seconds": 900,
    "idle_unload_seconds": 120
  }'
```

Then use:

- `POST /admin/api/runtime/model-pool/load`
- `POST /admin/api/runtime/model-pool/unload`
- `POST /admin/api/runtime/model-pool/pin`
- `GET /admin/api/runtime/model-pool`

## Telemetry

`/admin/api/runtime` now preserves the low-level GPU telemetry state instead of
silently hiding it. On this DGX Spark, if `nvidia-smi` fails with an NVML error,
you will see it under:

- `backend.details.telemetry.gpu_metrics_ok`
- `backend.details.telemetry.gpu_metrics_error`
- `backend.details.telemetry.system_memory_kb`

Phase 3 also adds session-restore diagnostics under:

- `backend.details.session_restore.last_save`
- `backend.details.session_restore.last_restore`
- `backend.details.session_restore.snapshots`

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
  --lmstudio-variant lmstudio,http://127.0.0.1:1234,qwen3.5-4b@q4_k_m \
  --target-context-tokens 65536 \
  --include-concurrency
```

The output now includes:

- `serving_preset`
- effective benchmark context fields
- telemetry health/error fields
- single-session follow-up latency
- managed restart recovery replay latency
- mixed-traffic makespan
- automatic recommendations for:
  - best single-session follow-up
  - best mixed-traffic profile
  - best long-output throughput
  - best cold long-prefix latency

## Codex CLI Phase Execution

The DGX roadmap is also split into phase task books for `Codex CLI`.

- roadmap index: `plans/codex-roadmap.md`
- phase docs: `plans/codex-phase-1.md` through `plans/codex-phase-5.md`
- repo-local skill source: `codex_skills/omlx-dgx-phase-executor/`
- launcher: `scripts/run_codex_phase.sh`

Run one phase at a time:

```bash
bash scripts/run_codex_phase.sh 1
```

Do not feed all five phases into a single Codex run.

For a real five-step sequential run, use:

```bash
bash scripts/run_codex_all_phases.sh
```

This runner executes Phase 1 through Phase 5 in order, saves each result under
`.runtime/codex-phase-runs/`, and stops immediately if any phase does not end
with `Phase Status: PASS`.

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
