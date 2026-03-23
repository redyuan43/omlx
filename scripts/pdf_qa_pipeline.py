#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Route PDF extraction and ask the configured chat model a question."""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pdf_route_extract import classify_pdf_route, extract_with_page_strategy
else:
    from scripts.pdf_route_extract import classify_pdf_route, extract_with_page_strategy


def _http_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def _chunk_text(text: str, *, max_chars: int = 1200) -> List[Dict[str, Any]]:
    normalized = text.replace("\f", "\n[page-break]\n")
    raw_parts = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    chunks: List[Dict[str, Any]] = []
    page = 1
    current = ""
    for part in raw_parts:
        if part == "[page-break]":
            if current:
                chunks.append({"page": page, "text": current.strip()})
                current = ""
            page += 1
            continue
        candidate = part if not current else f"{current}\n\n{part}"
        if current and len(candidate) > max_chars:
            chunks.append({"page": page, "text": current.strip()})
            current = part
        else:
            current = candidate
    if current:
        chunks.append({"page": page, "text": current.strip()})
    return [
        {"id": f"chunk_{index+1}", "page": chunk["page"], "text": chunk["text"]}
        for index, chunk in enumerate(chunks)
    ]


def _chunk_extraction(extraction: Dict[str, Any], *, max_chars: int = 1200) -> List[Dict[str, Any]]:
    if extraction.get("mode") != "page_auto":
        return _chunk_text(extraction["text"], max_chars=max_chars)

    chunks: List[Dict[str, Any]] = []
    next_id = 1
    for page_record in extraction.get("pages", []):
        page = page_record["page"]
        body_text = page_record.get("body_text", "").strip()
        if body_text:
            for body_chunk in _chunk_text(body_text, max_chars=max_chars):
                chunks.append(
                    {
                        "id": f"chunk_{next_id}",
                        "page": page,
                        "kind": "body_text",
                        "route": page_record.get("route", ""),
                        "text": body_chunk["text"],
                    }
                )
                next_id += 1
        image_caption = page_record.get("image_caption", "").strip()
        if image_caption:
            chunks.append(
                {
                    "id": f"chunk_{next_id}",
                    "page": page,
                    "kind": "image_caption",
                    "route": page_record.get("route", ""),
                    "text": image_caption,
                }
            )
            next_id += 1
    return chunks


def _retrieve_context(
    *,
    chat_url: str,
    embed_model: str,
    rerank_model: str,
    question: str,
    text: str | None = None,
    extraction: Dict[str, Any] | None = None,
    top_k: int = 8,
    rerank_top_n: int = 4,
) -> Dict[str, Any]:
    source = extraction if extraction is not None else {"text": text or ""}
    chunks = _chunk_extraction(source, max_chars=1200)
    if not chunks:
        return {"chunks": [], "selected": [], "context": ""}

    embeddings = _http_json(
        chat_url.rstrip("/") + "/v1/embeddings",
        {"model": embed_model, "input": [question] + [chunk["text"] for chunk in chunks]},
    )
    vectors = [item["embedding"] for item in embeddings["data"]]
    query_vec = vectors[0]
    scored = sorted(
        (
            {
                "id": chunk["id"],
                "page": chunk["page"],
                "kind": chunk.get("kind", "body_text"),
                "route": chunk.get("route", ""),
                "text": chunk["text"],
                "score": _cosine(query_vec, vector),
            }
            for chunk, vector in zip(chunks, vectors[1:])
        ),
        key=lambda item: item["score"],
        reverse=True,
    )
    top_chunks = scored[:top_k]
    rerank = _http_json(
        chat_url.rstrip("/") + "/v1/rerank",
        {
            "model": rerank_model,
            "query": question,
            "documents": [item["text"] for item in top_chunks],
        },
    )
    selected = [top_chunks[item["index"]] for item in rerank["results"][:rerank_top_n]]
    context = "\n\n---\n\n".join(
        f"[{item['id']} page={item['page']} kind={item.get('kind', 'body_text')} route={item.get('route', '')}]\n{item['text']}"
        for item in selected
    )
    return {
        "chunks": chunks,
        "selected": selected,
        "context": context,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PDF route, extract, and QA")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("question", help="Question to answer from the extracted text")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8008")
    parser.add_argument("--chat-model", default="qwen35-35b")
    parser.add_argument("--ocr-url", default="http://127.0.0.1:8012")
    parser.add_argument("--ocr-model", default="ocr-lite")
    parser.add_argument("--embed-model", default="embed-text")
    parser.add_argument("--rerank-model", default="rerank-qwen")
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--disable-rag", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--rerank-top-n", type=int, default=4)
    parser.add_argument("--strategy", default="extraction_rag")
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    route = classify_pdf_route(pdf_path)
    extraction = extract_with_page_strategy(
        pdf_path,
        chat_url=args.chat_url,
        chat_model=args.chat_model,
        ocr_url=args.ocr_url,
        ocr_model=args.ocr_model,
        first_page=args.first_page,
        last_page=args.last_page,
    )

    retrieval = None
    if args.disable_rag:
        body_text = extraction.get("text", "")
        caption_text = extraction.get("image_captions", "")
        context = f"{body_text}\n\n{caption_text}".strip()[: args.max_context_chars]
    else:
        retrieval = _retrieve_context(
            chat_url=args.chat_url,
            embed_model=args.embed_model,
            rerank_model=args.rerank_model,
            question=args.question,
            extraction=extraction,
            top_k=args.top_k,
            rerank_top_n=args.rerank_top_n,
        )
        context = retrieval["context"][: args.max_context_chars]
    prompt = (
        "请仅根据下面抽取文本回答问题。如果文本里没有答案，就明确说未找到。"
        f"\n\n问题：{args.question}\n\n抽取文本：\n{context}"
    )
    response = _http_json(
        args.chat_url.rstrip("/") + "/v1/chat/completions",
        {
            "model": args.chat_model,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 160,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    result = {
        "route": route,
        "extraction": extraction,
        "retrieval": retrieval,
        "qa_response": response,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
