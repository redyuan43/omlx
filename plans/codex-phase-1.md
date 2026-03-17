# Phase 1: Deepen The Current 4B GGUF Mainline

## Goal

Strengthen the current DGX Spark text-serving mainline without changing the model family:

- keep `Qwen3.5-4B GGUF + llama.cpp + Q4_K_M`
- preserve the existing wins on single-session follow-up and mixed traffic
- extend the benchmarked path from `32k` toward `64k`
- make session continuation and short-interruption recovery a first-class path

## Allowed Scope

You may modify only these areas:

- `omlx_dgx/runtime/llama_cpp.py`
- `omlx_dgx/control_plane/`
- `omlx_dgx/config.py`
- `omlx_dgx/cli.py`
- `scripts/bench_qwen35_4b.py`
- `scripts/bench_qwen35_4b_concurrency.py`
- `scripts/bench_qwen35_4b_matrix.py`
- `tests/test_omlx_dgx_llama_cpp_adapter.py`
- `tests/test_omlx_dgx_config.py`
- `tests/test_omlx_dgx_app.py`
- `README.dgx.md`
- `README.md`

## Out Of Scope

Do not do any of the following in this phase:

- switch the mainline back to `SGLang`, `AWQ`, or `TensorRT-LLM`
- add multi-model pool management
- add embeddings, rerank, or VLM/OCR
- change external API shapes beyond what the current control plane already supports
- mix in work from Phase 2 through Phase 5

## Required Work

1. Add a recoverable single-session continuation path suitable for short restarts or brief runtime interruptions.
2. Push the current single-session benchmark path from `32k` to `64k` where feasible.
3. Keep `Q4_K_M` as the default GGUF recommendation unless new measurements prove otherwise.
4. Preserve or improve the current follow-up wins against `LM Studio`.
5. Preserve the current mixed-traffic safety net so that unrelated short requests do not blow away a warm long-context slot.

## Required Validation

Run these tests:

```bash
PYTHONPATH=. .venv/bin/pytest -q --noconftest \
  tests/test_omlx_dgx_config.py \
  tests/test_omlx_dgx_llama_cpp_adapter.py \
  tests/test_omlx_dgx_app.py
```

Run these benchmarks against real services and report the actual URLs used:

```bash
python3 scripts/bench_qwen35_4b.py \
  --control-plane-url http://127.0.0.1:PORT \
  --runtime-url http://127.0.0.1:PORT

python3 scripts/bench_qwen35_4b_concurrency.py \
  --control-plane-url http://127.0.0.1:PORT \
  --runtime-url http://127.0.0.1:PORT

python3 scripts/bench_lmstudio_qwen35_4b.py \
  --lmstudio-url http://127.0.0.1:1234 \
  --model qwen3.5-4b
```

## Success Criteria

- single-session follow-up remains stronger than `LM Studio` on the same `Q4_K_M` path
- `64k` runs do not regress into unusable latency or failures
- repeat long-prefix reuse remains visible and measurable
- mixed-traffic protection still works after the continuation/recovery changes
- README benchmark guidance remains accurate

## Required Output

End with:

1. `Changes`
2. `Benchmarks`
3. `Known Issues`
4. `Next-Phase Recommendation`
