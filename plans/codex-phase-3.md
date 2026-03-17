# Phase 3: Add DGX Hot And Cold Session Restore

## Goal

Bring the DGX path closer to the upstream hot+cold cache idea without pretending that GGUF already has native block-level KV restore.

Phase 3 should add:

- hot session state based on runtime slot/checkpoint/continuation
- cold session metadata snapshots
- restart-aware restore that is meaningfully faster than a fully cold prefill

## Allowed Scope

- `omlx_dgx/runtime/llama_cpp.py`
- `omlx_dgx/tiered_kv.py`
- `omlx_dgx/control_plane/`
- `omlx_dgx/config.py`
- snapshot/manifest helpers under `omlx_dgx/`
- benchmark scripts and tests
- `README.dgx.md`
- `README.md`

## Out Of Scope

- no claim of true GGUF block-level KV restore unless it is actually implemented and benchmarked
- no embeddings, rerank, or VLM/OCR
- no later-phase interface work

## Required Work

1. Persist session-level restore metadata for warm long-context chats.
2. Support restore after short runtime restarts.
3. Surface restore status and timing in admin diagnostics.
4. Compare restored follow-up latency against cold prefill latency.

## Required Validation

- run real restart-and-restore benchmarks
- show cold-start versus restored session timings
- keep the existing single-session follow-up and mixed-traffic paths working

## Success Criteria

- restored sessions are measurably faster than cold-started sessions
- snapshot metadata is visible and test-covered
- restore failures degrade safely back to cold execution

## Required Output

End with:

1. `Changes`
2. `Benchmarks`
3. `Known Issues`
4. `Next-Phase Recommendation`
