# Jetson Local Service Guide

This document is the current local-usage guide for the Jetson AGX Orin setup in
this repository. It reflects the stable local topology that is actually running
on this machine now.

Scope:

- main text and vision-chat service: `Qwen3.5-35B` on `oMLX + llama.cpp`
- embeddings: external OpenAI-compatible embedding model
- rerank: managed `llama.cpp` reranker
- OCR: separate `oMLX + llama.cpp` OCR service
- named disk-backed contexts: supported on the main chat path

Important distinction:

- `Qwen3.5-35B` as a model family can be multimodal.
- The current local alias `qwen35-35b` on port `8008` is now deployed as a
  single-runtime text + vision-chat model in this setup.
- OCR still stays on the separate `ocr-lite` service.

## 1. Current Local Topology

| Purpose | Base URL | Model Alias | Notes |
| --- | --- | --- | --- |
| Main control plane | `http://127.0.0.1:8008` | `qwen35-35b` | Default text and vision-chat model |
| Embeddings via main control plane | `http://127.0.0.1:8008` | `embed-text` | Proxied to local Ollama-compatible embeddings backend |
| Rerank via main control plane | `http://127.0.0.1:8008` | `rerank-qwen` | Managed `llama.cpp --reranking` model |
| OCR control plane | `http://127.0.0.1:8012` | `ocr-lite` | Dedicated OCR service |
| Main llama.cpp runtime | `http://127.0.0.1:30000` | internal | Debug only, do not call directly in normal use |
| Rerank llama.cpp runtime | `http://127.0.0.1:30030` | internal | Managed by the main control plane |
| OCR llama.cpp runtime | `http://127.0.0.1:30020` | internal | Managed by the OCR control plane |
| External embeddings backend | `http://127.0.0.1:11434` | `nomic-embed-text` | Required for `/v1/embeddings` on `8008` |

Stable main text configuration:

- model: `Qwen3.5-35B-A3B.Q4_K_M.gguf`
- `ctx_size=32768`
- `parallel_slots=1`

Current multimodal attachment:

- `mmproj_path=/data/models/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B.mmproj-Q8_0.gguf`
- the model is registered with `primary_service=chat`
- `supports_vision=true`
- `supports_ocr=false`
- OCR is split to the dedicated `ocr-lite` service on `8012`

## 2. How To Run The Services

### 2.1 Prerequisites

The current local setup expects:

- Python venv: `/data/venvs/omlx-dgx`
- built `llama.cpp` server:
  - `/data/src/llama.cpp/build/bin/llama-server`
- main model:
  - `/data/models/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B.Q4_K_M.gguf`
- rerank model:
  - `/data/models/Qwen3-Reranker-0.6B-Q8_0-GGUF/qwen3-reranker-0.6b-q8_0.gguf`
- OCR model:
  - `/data/models/GLM-OCR-GGUF/GLM-OCR.Q4_K_M.gguf`
  - `/data/models/GLM-OCR-GGUF/GLM-OCR.mmproj-Q8_0.gguf`
- local embeddings backend on port `11434` with:
  - `nomic-embed-text`

If you want embeddings and RAG, make sure the embeddings backend is running:

```bash
ollama serve
ollama pull nomic-embed-text
```

### 2.2 Start The Main 35B Service

```bash
/data/venvs/omlx-dgx/bin/python -m omlx_dgx.cli serve \
  --base-path /data/omlx-dgx-35b-ctx32768
```

This starts:

- control plane on `127.0.0.1:8008`
- managed `qwen35-35b` runtime on `127.0.0.1:30000`
- managed `rerank-qwen` runtime on `127.0.0.1:30030` when needed

### 2.3 Start The OCR Service

```bash
/data/venvs/omlx-dgx/bin/python -m omlx_dgx.cli serve \
  --base-path /data/omlx-dgx-ocr-lite-gguf
```

This starts:

- OCR control plane on `127.0.0.1:8012`
- managed `ocr-lite` runtime on `127.0.0.1:30020`

### 2.4 Verify

```bash
curl http://127.0.0.1:8008/v1/models
curl http://127.0.0.1:8008/admin/api/runtime
curl http://127.0.0.1:8012/v1/models
```

## 3. Can A Frontend Use Standard OpenAI Or Anthropic APIs?

Yes, with two practical notes:

1. The service is endpoint-shape compatible with standard OpenAI and Anthropic
   APIs.
2. Model capability is model-scoped. Frontends must select the correct model
   alias for the task.

Supported endpoint shapes on the main service at `8008`:

| Endpoint | Compatibility | Use |
| --- | --- | --- |
| `POST /v1/chat/completions` | OpenAI-style | chat |
| `POST /v1/completions` | OpenAI-style | text completion |
| `POST /v1/messages` | Anthropic Messages API | chat |
| `POST /v1/embeddings` | OpenAI-style | embeddings |
| `POST /v1/rerank` | repo endpoint | rerank |
| `GET /v1/models` | OpenAI-style | list models |

Important limitations:

- `qwen35-35b` on `8008` now supports both normal text chat and vision-language
  image chat.
- OCR is currently a separate service on `8012`.
- `embed-text` cannot answer chat requests.
- `rerank-qwen` cannot answer chat or embedding requests.
- `ocr-lite` should be used only for OCR requests.

Browser frontend note:

- The current FastAPI app does not install CORS middleware.
- If your frontend is a browser app on another origin, use a small same-origin
  proxy or backend relay instead of direct browser-to-`8008` calls.

## 4. Model Aliases You Should Use

| Alias | Service | Where | Notes |
| --- | --- | --- | --- |
| `qwen35-35b` | `chat` + `vision_chat` | `8008` | single-runtime text and image chat |
| `embed-text` | `embeddings` | `8008` | proxied to local embeddings backend |
| `rerank-qwen` | `rerank` | `8008` | managed reranker |
| `ocr-lite` | `ocr` | `8012` | dedicated OCR model |

Do not rely on "first model in `/v1/models`". Pick the correct alias explicitly.

## 5. OpenAI-Compatible Usage

### 5.1 Chat

```bash
curl http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-35b",
    "stream": false,
    "messages": [
      {"role": "user", "content": "Reply with exactly PONG."}
    ],
    "extra_body": {"think": false}
  }'
```

JavaScript example:

```javascript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8008/v1",
  apiKey: "local"
});

const res = await client.chat.completions.create({
  model: "qwen35-35b",
  messages: [{ role: "user", content: "Reply with exactly PONG." }],
  extra_body: { think: false }
});

console.log(res.choices[0].message.content);
```

### 5.2 Named Disk-Backed Contexts

Use `metadata.context_id` to bind requests to a named context that can be
restored after runtime restart.

First request creates and saves the context:

```bash
curl http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-35b",
    "stream": false,
    "metadata": {"context_id": "repo-a"},
    "messages": [
      {
        "role": "user",
        "content": "This is the long base context. Keep it for later requests."
      }
    ],
    "extra_body": {"think": false}
  }'
```

Later requests reuse the same `context_id`:

```bash
curl http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-35b",
    "stream": false,
    "metadata": {"context_id": "repo-a"},
    "messages": [
      {"role": "user", "content": "Continue from the saved context and answer this follow-up."}
    ],
    "extra_body": {"think": false}
  }'
```

Admin context management:

```bash
curl http://127.0.0.1:8008/admin/api/contexts
curl http://127.0.0.1:8008/admin/api/contexts/repo-a
curl -X POST http://127.0.0.1:8008/admin/api/contexts/repo-a/restore
curl -X DELETE http://127.0.0.1:8008/admin/api/contexts/repo-a
```

### 5.3 Embeddings

```bash
curl http://127.0.0.1:8008/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "embed-text",
    "input": ["hello world", "jetson local rag"]
  }'
```

### 5.4 Rerank

`/v1/rerank` is supported by the service, but it is not an official OpenAI SDK
method. Use raw HTTP or `fetch`.

```bash
curl http://127.0.0.1:8008/v1/rerank \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "rerank-qwen",
    "query": "Which paragraph explains OCR routing?",
    "documents": [
      "Chat requests go to qwen35-35b.",
      "OCR requests should go to ocr-lite on the OCR service.",
      "Embeddings are served by embed-text."
    ],
    "top_n": 2,
    "return_documents": true
  }'
```

## 6. Anthropic-Compatible Usage

The main chat service supports the Anthropic Messages API shape at:

- `POST http://127.0.0.1:8008/v1/messages`

Example:

```bash
curl http://127.0.0.1:8008/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-35b",
    "max_tokens": 64,
    "messages": [
      {"role": "user", "content": "Reply with exactly PONG."}
    ]
  }'
```

JavaScript example:

```javascript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({
  baseURL: "http://127.0.0.1:8008",
  apiKey: "local"
});

const res = await client.messages.create({
  model: "qwen35-35b",
  max_tokens: 64,
  messages: [{ role: "user", content: "Reply with exactly PONG." }]
});

console.log(res.content);
```

Use Anthropic-style requests only for chat/messages. Embeddings, rerank, and OCR
still use the OpenAI-style or raw HTTP routes described above.

## 7. Vision Chat Usage

The main `qwen35-35b` service on `8008` now accepts image inputs.

Example:

```bash
curl http://127.0.0.1:8008/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen35-35b",
    "stream": false,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What kind of scene is shown in this image? Answer briefly."},
          {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,<BASE64_IMAGE>"}
          }
        ]
      }
    ],
    "extra_body": {"think": false}
  }'
```

Current boundary:

- image chat: `qwen35-35b` on `8008`
- OCR extraction: `ocr-lite` on `8012`

## 8. OCR Usage

OCR is currently a separate service on `8012`. Use the OCR model alias
`ocr-lite`.

Request shape:

- endpoint: `POST /v1/chat/completions`
- model: `ocr-lite`
- OCR mode: set either:
  - top-level `"ocr": true`, or
  - `"metadata": {"task": "ocr"}`

Example:

```bash
curl http://127.0.0.1:8012/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ocr-lite",
    "stream": false,
    "ocr": true,
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract all visible text."},
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64,<BASE64_IMAGE>"
            }
          }
        ]
      }
    ]
  }'
```

Notes:

- `ocr-lite` is an OCR model, not a normal chat model.
- In the current local topology, OCR is still not routed through the main
  `8008` service.

## 9. RAG Usage

RAG is a composed workflow, not a single endpoint.

Current local stack:

1. embed with `embed-text`
2. retrieve in your application
3. rerank with `rerank-qwen`
4. answer with `qwen35-35b`

For PDFs, there is already a helper pipeline in:

- [scripts/pdf_qa_pipeline.py](../scripts/pdf_qa_pipeline.py)

Example:

```bash
python scripts/pdf_qa_pipeline.py \
  "/path/to/file.pdf" \
  "What is the document title?" \
  --chat-url http://127.0.0.1:8008 \
  --chat-model qwen35-35b \
  --ocr-url http://127.0.0.1:8012 \
  --ocr-model ocr-lite \
  --embed-model embed-text \
  --rerank-model rerank-qwen
```

This script does:

- page-by-page PDF route decision
- `direct_text` for clean selectable text
- `ocr_text` for noisy or scanned text-heavy pages
- `ocr_text_plus_vlm_caption` for mixed pages with charts/figures
- `vlm_only` for mostly visual pages
- embedding + retrieval + rerank
- final answer with `qwen35-35b`

For RAG, the pipeline now keeps:

- `body_text` as the primary retrievable text
- `image_caption` as a separate retrievable visual description

## 10. PDF Routing And Extraction

Helper script:

- [scripts/pdf_route_extract.py](../scripts/pdf_route_extract.py)

Example:

```bash
python scripts/pdf_route_extract.py \
  "/path/to/file.pdf" \
  --chat-url http://127.0.0.1:8008 \
  --chat-model qwen35-35b \
  --ocr-url http://127.0.0.1:8012 \
  --ocr-model ocr-lite
```

This script now uses a page-level extraction strategy for RAG and chooses between:

- `direct_text`
- `ocr_text`
- `ocr_text_plus_vlm_caption`
- `vlm_only`

The current local recommendation is:

- use `direct_text` if the page already has a clean text layer
- use `ocr_text` if the page is scanned or the text layer is noisy
- use `ocr_text_plus_vlm_caption` when OCR should extract the text but `qwen35-35b`
  should also explain the chart, figure, or image
- use `vlm_only` only for pages whose main value is visual understanding rather than
  text extraction

The main design rule is:

- `qwen35-35b` multimodal is not the primary text extractor
- OCR or direct text stays the source of `body_text`
- multimodal output is stored separately as `image_caption`

System binaries required by the helper scripts:

- `pdfinfo`
- `pdftotext`
- `pdftoppm`

## 11. Admin And Diagnostics

Useful admin endpoints on the main service:

```bash
curl http://127.0.0.1:8008/admin/api/runtime
curl http://127.0.0.1:8008/admin/api/runtime/capabilities
curl http://127.0.0.1:8008/admin/api/runtime/model-pool
curl http://127.0.0.1:8008/admin/api/runtime/cache-report
curl http://127.0.0.1:8008/admin/api/contexts
curl http://127.0.0.1:8008/admin/api/benchmarks
```

Useful admin endpoints on the OCR service:

```bash
curl http://127.0.0.1:8012/admin/api/runtime
curl http://127.0.0.1:8012/v1/models
```

## 12. Practical Notes

- Use `qwen35-35b` explicitly for chat. Do not let a generic frontend
  accidentally pick `embed-text`.
- `qwen35-35b` is now the single-runtime text + vision-chat model on `8008`.
- Keep OCR requests on `ocr-lite` unless you are intentionally testing
  multimodal question answering instead of OCR extraction.
- The stable 35B production path is currently:
  - `ctx_size=32768`
  - `parallel_slots=1`
- `parallel_slots=2` is not the current stable default for 35B.
- The main service is text + vision-chat. OCR remains split out into the
  dedicated `8012` service.
- If embeddings are failing, check the external embeddings backend at `11434`
  first.
- If an SDK requires an API key, use a placeholder such as `local`.

## 13. Where To Look In The Repo

- main DGX overview: [README.dgx.md](../README.dgx.md)
- main service settings:
  - `/data/omlx-dgx-35b-ctx32768/settings.json`
- OCR service settings:
  - `/data/omlx-dgx-ocr-lite-gguf/settings.json`
- PDF route helper:
  - [scripts/pdf_route_extract.py](../scripts/pdf_route_extract.py)
- PDF QA helper:
  - [scripts/pdf_qa_pipeline.py](../scripts/pdf_qa_pipeline.py)

## 14. Current Deployment Decision And Tradeoff

The current local deployment is now:

- one `qwen35-35b` runtime for text + vision-chat on `8008`
- one separate `ocr-lite` runtime for OCR on `8012`

This keeps the current tradeoff:

- image chat uses the same 35B runtime
- OCR extraction still uses the specialized OCR model

Why OCR is still separate:

- OCR extraction quality is still better on the dedicated OCR path
- OCR traffic should not become the default load on the main 35B service
- routing stays cleaner:
  - image chat -> `qwen35-35b`
  - OCR extraction -> `ocr-lite`
