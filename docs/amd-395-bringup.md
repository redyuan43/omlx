# AMD 395 AI+ Bring-up

This document brings the current DGX baseline onto an `AMD RYZEN AI MAX+ 395 w/ Radeon 8060S`.

## Target Baseline

- mainline: `Qwen3.5-4B GGUF + llama.cpp + Q4_K_M`
- default preset: `single_session_low_latency`
- mixed preset: `mixed_traffic`
- context: `32768`
- acceptance: same benchmark and smoke scripts as the current DGX path

## Prerequisites

- ROCm is installed and `rocminfo`, `rocm-smi`, `hipconfig` work
- the current user can access `/dev/kfd` and the `render` group
- LM Studio is installed for the auxiliary `embedding / vision / OCR` providers
- LM Studio local server is enabled on `http://127.0.0.1:1234`
- LM Studio has these models loaded:
  - one `Qwen3-VL-4B` model
  - one `GLM-OCR` model
  - one `text-embedding-nomic-embed-text-v1.5` model

If `lms` hangs on `Waking up LM Studio service...`, check
`"~/.lmstudio/.internal/app-install-location.json"`. A stale AppImage mount such as
`"/tmp/.mount_..."` will break the daemon wake-up path. Point it back to the current
install, for example:

```json
{"path":"/opt/LM Studio/lm-studio","argv":["/opt/LM Studio/lm-studio"],"cwd":"/opt/LM Studio"}
```

## 1. Build HIP llama.cpp

```bash
bash "scripts/build_llama_cpp_rocm.sh" --gpu-targets gfx1151
```

Default output:

- source: `"/home/ivan/github/omlx/.runtime/llama.cpp-src"`
- build: `"/home/ivan/github/omlx/.runtime/llama.cpp-build"`
- binary: `"/home/ivan/github/omlx/.runtime/llama.cpp-build/bin/llama-server"`

If your ROCm stack needs a different target, pass `--gpu-targets`.

## 2. Download the GGUF artifacts

```bash
python3 "scripts/download_amd_baseline_models.py" \
  --output-dir "~/.cache/omlx-dgx-models"
```

This downloads:

- `lmstudio-community/Qwen3.5-4B-GGUF` `Q4_K_M`
- `ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF`

The script prints the resolved local paths as JSON.

## 3. Start the single-session baseline

```bash
bash "scripts/run_amd_qwen35_baseline.sh" \
  --artifact-path "~/.cache/omlx-dgx-models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
```

Default endpoints:

- control plane: `http://127.0.0.1:8010`
- runtime: `http://127.0.0.1:31000`
- state root: `~/.omlx-dgx-amd/qwen35-4b-single`

If the integrated GPU path only comes up with unified memory enabled:

```bash
bash "scripts/run_amd_qwen35_baseline.sh" \
  --artifact-path "~/.cache/omlx-dgx-models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf" \
  --enable-uma-fallback
```

## 4. Start the mixed-traffic baseline

```bash
bash "scripts/run_amd_qwen35_baseline.sh" \
  --preset mixed_traffic \
  --artifact-path "~/.cache/omlx-dgx-models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf"
```

Default endpoints:

- control plane: `http://127.0.0.1:8020`
- runtime: `http://127.0.0.1:31200`
- state root: `~/.omlx-dgx-amd/qwen35-4b-mixed`

## 5. Register rerank + LM Studio capability models

Single-session control plane:

```bash
python3 "scripts/register_amd_baseline_models.py" \
  --control-plane-url "http://127.0.0.1:8010" \
  --lmstudio-url "http://127.0.0.1:1234" \
  --rerank-artifact-path "~/.cache/omlx-dgx-models/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf" \
  --rerank-base-url "http://127.0.0.1:31020"
```

Mixed-traffic control plane:

```bash
python3 "scripts/register_amd_baseline_models.py" \
  --control-plane-url "http://127.0.0.1:8020" \
  --lmstudio-url "http://127.0.0.1:1234" \
  --rerank-artifact-path "~/.cache/omlx-dgx-models/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf" \
  --rerank-base-url "http://127.0.0.1:31220"
```

Registered models:

- `qwen35-reranker` via managed `llama.cpp`
- `nomic-embed` via LM Studio
- `qwen3-vl-4b` via LM Studio
- `glm-ocr` via LM Studio

If LM Studio is still unavailable, you can register the local reranker first and leave
`embedding / vision / OCR` for a later pass:

```bash
python3 "scripts/register_amd_baseline_models.py" \
  --control-plane-url "http://127.0.0.1:8010" \
  --rerank-artifact-path "~/.cache/omlx-dgx-models/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf" \
  --skip-lmstudio
```

## 6. Validate the baseline

Single-session benchmark:

```bash
python3 "scripts/bench_qwen35_4b.py" \
  --control-plane-url "http://127.0.0.1:8010" \
  --runtime-url "http://127.0.0.1:31000"
```

Mixed-traffic benchmark:

```bash
python3 "scripts/bench_qwen35_4b_concurrency.py" \
  --control-plane-url "http://127.0.0.1:8020" \
  --runtime-url "http://127.0.0.1:31200"
```

Capability smoke:

```bash
python3 "scripts/bench_multimodal_smoke.py" \
  --control-plane-url "http://127.0.0.1:8010"
```

Success means:

- `bench_qwen35_4b.py` returns a complete JSON report
- `bench_qwen35_4b_concurrency.py` returns a complete JSON report
- `bench_multimodal_smoke.py` reports `embeddings_ok=true`, `rerank_ok=true`, `vision_ok=true`, `ocr_ok=true`
- if you are on the rerank-only fallback path, `/v1/rerank` should already return `200 OK`

## Notes

- The current repo keeps the main Qwen chat path on managed `llama.cpp`.
- `embedding / vision / OCR` stay on externally registered OpenAI-compatible providers.
- `rocm-smi` telemetry is surfaced in `/admin/api/runtime` when the active backend is not on NVIDIA.
