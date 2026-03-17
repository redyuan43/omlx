# Codex CLI DGX Roadmap

This roadmap is a long-lived execution contract for `/home/dgx/github/omlx`.

Do not execute all phases in one run.

Rules:

- Execute exactly one phase per Codex session.
- Finish the current phase before proposing the next one.
- Keep the DGX mainline on `Qwen3.5-4B GGUF + llama.cpp + Q4_K_M` unless a phase explicitly changes that.
- Benchmark and report every phase before closing it.
- Mark every phase result with `Phase Status: PASS` or `Phase Status: FAIL`.

Recommended order:

1. `Phase 1`: deepen the current 4B GGUF mainline
2. `Phase 2`: multi-model pin/TTL/LRU/load-unload
3. `Phase 3`: DGX hot+cold session restore
4. `Phase 4`: `/v1/messages`, embeddings, rerank, benchmark endpoints
5. `Phase 5`: VLM and OCR

Execution helper:

```bash
bash scripts/run_codex_phase.sh 1
```

Fully automated five-phase runner:

```bash
bash scripts/run_codex_all_phases.sh
```

If a phase is blocked, stop after reporting:

- what changed
- benchmark results
- known issues
- whether the next phase is recommended
