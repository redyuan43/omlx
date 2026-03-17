---
name: omlx-dgx-phase-executor
description: Execute one DGX Spark optimization phase at a time in /home/dgx/github/omlx using Codex CLI. Use when implementing, benchmarking, or reviewing the repo task books under plans/codex-phase-*.md. Never execute more than one phase in a single run; always honor the phase boundaries, required benchmarks, and stop conditions.
---

# oMLX DGX Phase Executor

Use this skill only for `/home/dgx/github/omlx`.

## Start Here

1. Read `README.dgx.md` for the current DGX Spark baseline.
2. Read exactly one phase document under `plans/codex-phase-N.md`.
3. Read `references/current-baseline.md` when the phase needs benchmark context.

## Execution Contract

- Execute one phase per run.
- Stay inside the allowed file paths listed in the phase document.
- Do not pull work forward from later phases.
- Keep the mainline on `Qwen3.5-4B GGUF + llama.cpp + Q4_K_M` unless the phase document explicitly says otherwise.
- Do not switch the DGX mainline back to `AWQ + SGLang` or `TensorRT-LLM`.
- Do not reintroduce Apple/macOS menubar logic into the DGX path.
- Run the required tests and benchmarks from the phase document before declaring completion.
- Report benchmark numbers and known issues explicitly.

## Preferred Repo Assets

- DGX baseline and measured results: `README.dgx.md`
- Runtime and control plane code: `omlx_dgx/`
- Benchmark scripts:
  - `scripts/bench_qwen35_4b.py`
  - `scripts/bench_qwen35_4b_concurrency.py`
  - `scripts/bench_qwen35_4b_matrix.py`
  - `scripts/bench_lmstudio_qwen35_4b.py`
- Phase launcher helper: `scripts/run_codex_phase.sh`

## Required Output

End every phase run with these four sections:

1. `Changes`
2. `Benchmarks`
3. `Known Issues`
4. `Next-Phase Recommendation`

Then append a final status line:

- `Phase Status: PASS`
- or `Phase Status: FAIL`

If the phase is blocked, stop and explain which acceptance criteria were not met.

## Phase Order

- `Phase 1`: deepen the current 4B GGUF mainline
- `Phase 2`: add multi-model pin/TTL/LRU/load-unload control
- `Phase 3`: add DGX hot+cold session restore
- `Phase 4`: add `/v1/messages`, embeddings, rerank, and benchmark endpoints
- `Phase 5`: add VLM and OCR last

Never skip directly to a later phase unless the user explicitly overrides the order.

## Stop Conditions

Stop the run and report instead of continuing when:

- the phase acceptance criteria are not met
- the next step would require another phase's scope
- benchmark regressions exceed the current phase budget
- the required runtime or model artifacts are unavailable
