# Phase 5: Add VLM And OCR Last

## Goal

Add multimodal coverage only after the text path is already stable:

- VLM
- OCR

The text-serving performance path must remain the priority.

## Allowed Scope

- model capability wiring
- control-plane routing
- DGX runtime adapters as needed
- tests
- benchmark docs
- `README.dgx.md`
- `README.md`

## Out Of Scope

- no regression of the text-serving default path
- no uncontrolled scope expansion into unrelated runtime research

## Required Work

1. Introduce explicit multimodal capabilities and routing.
2. Add benchmark and smoke-test coverage for VLM/OCR.
3. Keep text workloads isolated from multimodal regressions.

## Required Validation

- text benchmarks still pass
- multimodal requests route only to supported backends
- admin diagnostics declare model capabilities clearly

## Success Criteria

- multimodal support exists without weakening the text mainline
- capability reporting is clear
- benchmark expectations are documented

## Required Output

End with:

1. `Changes`
2. `Benchmarks`
3. `Known Issues`
4. `Next-Phase Recommendation`
