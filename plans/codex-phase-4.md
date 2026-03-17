# Phase 4: Add General Service Interfaces

## Goal

Extend the DGX control plane beyond chat-only serving by adding clear service capabilities and benchmark surfaces:

- `/v1/messages`
- `/v1/embeddings`
- `/v1/rerank`
- benchmark output through a stable interface instead of scripts only

## Allowed Scope

- `omlx_dgx/control_plane/`
- `omlx_dgx/runtime/`
- `omlx_dgx/config.py`
- tests
- benchmark tooling
- `README.dgx.md`
- `README.md`

## Out Of Scope

- no VLM/OCR yet
- no unrelated model-pool redesign unless Phase 2 requires a small compatibility change
- no switch away from the current GGUF mainline

## Required Work

1. Define capability-aware routing and error handling for unsupported backends.
2. Add the new endpoints or explicit capability responses.
3. Expose benchmark reports through a stable API or admin task path.
4. Keep the chat path and DGX benchmark baselines intact.

## Required Validation

- endpoint tests for success and unsupported-capability behavior
- at least one real benchmark retrieval through the new interface
- no regression on existing chat benchmarks

## Success Criteria

- capabilities are explicit rather than implied
- unsupported features fail clearly
- benchmark reports can be consumed without manually running a local script

## Required Output

End with:

1. `Changes`
2. `Benchmarks`
3. `Known Issues`
4. `Next-Phase Recommendation`
