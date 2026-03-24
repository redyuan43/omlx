# AMD 395 AGX Parity Guide

This guide mirrors the `agx` branch topology on the current AMD 395 AI+ host.

Target topology:

- main control plane: `http://127.0.0.1:8008`
- main runtime: `http://127.0.0.1:30000`
- main model alias: `qwen35-35b`
- embeddings alias on main control plane: `embed-text`
- rerank alias on main control plane: `rerank-qwen`
- rerank runtime: `http://127.0.0.1:30030`
- OCR control plane: `http://127.0.0.1:8012`
- OCR runtime: `http://127.0.0.1:30020`
- OCR model alias: `ocr-lite`

## 1. Build llama.cpp ROCm

```bash
bash "scripts/build_llama_cpp_rocm.sh"
```

Default output:

- `"/home/ivan/github/omlx/.runtime/llama.cpp-build/bin/llama-server"`

## 2. Resolve Or Download Assets

The bootstrap script scans the usual local caches first:

- `~/.lmstudio/models`
- `~/.cache/omlx-dgx-models`

It can also download the missing rerank/OCR GGUF assets and pull the Ollama embedding model.

```bash
python3 "scripts/bootstrap_amd_agx_parity.py" \
  --output ".runtime/amd-agx-parity-manifest.json"
```

If OCR or embedding is still missing:

```bash
python3 "scripts/bootstrap_amd_agx_parity.py" \
  --download-missing \
  --pull-embedding \
  --output ".runtime/amd-agx-parity-manifest.json"
```

## 3. Start The Main Service

```bash
bash "scripts/run_amd_agx_parity_main.sh" \
  --manifest ".runtime/amd-agx-parity-manifest.json"
```

Or start the whole stack in the background:

```bash
bash "scripts/run_amd_agx_parity_stack.sh" \
  --manifest ".runtime/amd-agx-parity-manifest.json"
```

## 4. Register Embeddings And Rerank

```bash
python3 "scripts/register_amd_agx_parity_models.py" \
  --manifest ".runtime/amd-agx-parity-manifest.json"
```

## 5. Start The OCR Service

```bash
bash "scripts/run_amd_agx_parity_ocr.sh" \
  --manifest ".runtime/amd-agx-parity-manifest.json"
```

## 6. Verify

```bash
curl "http://127.0.0.1:8008/v1/models"
curl "http://127.0.0.1:8008/admin/api/runtime"
curl "http://127.0.0.1:8008/admin/api/contexts"
curl "http://127.0.0.1:8012/v1/models"
```

Main control plane should expose:

- `qwen35-35b`
- `embed-text`
- `rerank-qwen`

OCR control plane should expose:

- `ocr-lite`

## 7. Smoke Commands

One-shot strict parity smoke:

```bash
python3 "scripts/smoke_amd_agx_parity.py"
```

This checks:

- main models on `8008`
- OCR models on `8012`
- text chat
- named contexts
- embeddings
- rerank
- main-service vision
- dedicated OCR

If the main multimodal model does not support `llama.cpp` slot save, named contexts
fall back to `payload_replay`. This still keeps the AGX context management contract
alive on AMD, and `/admin/api/contexts` will report `storage_mode: payload_replay`.

Text chat:

```bash
curl "http://127.0.0.1:8008/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-35b",
    "stream": false,
    "messages": [{"role": "user", "content": "Reply with exactly PONG."}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Named context:

```bash
curl "http://127.0.0.1:8008/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35-35b",
    "stream": false,
    "metadata": {"context_id": "repo-a"},
    "messages": [{"role": "user", "content": "Remember this base context."}]
  }'
```

Embeddings:

```bash
curl "http://127.0.0.1:8008/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "embed-text",
    "input": "hello"
  }'
```

Rerank:

```bash
curl "http://127.0.0.1:8008/v1/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-qwen",
    "query": "hello",
    "documents": ["doc-a", "doc-b"]
  }'
```

OCR:

```bash
curl "http://127.0.0.1:8012/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ocr-lite",
    "stream": false,
    "ocr": true,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Read all text in this image."},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}
      ]
    }]
  }'
```

PDF/RAG helpers restored from `agx`:

- `python3 "scripts/pdf_route_extract.py" --help`
- `python3 "scripts/pdf_qa_pipeline.py" --help`
- `python3 "scripts/bench_pdf_route_strategy.py" --help`

## 8. Background Launcher

Status:

```bash
bash "scripts/run_amd_agx_parity_stack.sh" --status
```

Stop:

```bash
bash "scripts/stop_amd_agx_parity_stack.sh"
```

The stack launcher stores logs, pid files, smoke output, and status JSON under:

- `".runtime/amd-agx-parity-stack"`

## 9. User-Level systemd Templates

Template files live in:

- `ops/systemd/user/omlx-agx-main.service`
- `ops/systemd/user/omlx-agx-ocr.service`
- `ops/systemd/user/README.md`
