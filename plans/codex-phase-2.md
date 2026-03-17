# Phase 2: Add Multi-Model Serving Control

## Goal

Add DGX-side model-pool control that matches the spirit of the upstream README:

- multi-model registration
- pin
- TTL
- idle unload
- LRU eviction
- manual load/unload

This phase should start only after Phase 1 has benchmarked and stabilized the current 4B mainline.

## Allowed Scope

- `omlx_dgx/control_plane/`
- `omlx_dgx/config.py`
- `omlx_dgx/runtime/llama_cpp.py`
- new helper modules under `omlx_dgx/`
- tests covering config, control plane, and llama.cpp adapter behavior
- `README.dgx.md`
- `README.md`

## Out Of Scope

- no hot+cold restore store yet
- no embeddings, rerank, or VLM/OCR
- no switch away from `Q4_K_M` mainline
- no work from Phase 3 through Phase 5

## Required Work

1. Introduce a model-pool abstraction for GGUF text models.
2. Support pin, TTL, idle unload, LRU eviction, and manual load/unload.
3. Ensure second-model activity does not destroy the warm state of the primary long-context chat.
4. Add admin diagnostics showing loaded models, pin state, TTL, last used time, and eviction reasons.

## Required Validation

Run the relevant pytest subsets and at least one real benchmark showing:

- a primary long-context session remains usable
- a second model can load and unload
- pool actions are visible in admin diagnostics

## Success Criteria

- multiple GGUF models can be managed without breaking the primary long-context route
- pin, TTL, and LRU decisions are visible and test-covered
- the control plane reports clear model-pool state

## Required Output

End with:

1. `Changes`
2. `Benchmarks`
3. `Known Issues`
4. `Next-Phase Recommendation`
