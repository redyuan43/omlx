#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare OCR+35B vs direct 35B multimodal on a PDF, page by page."""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pdf_route_extract import _ocr_image, _pdf_pages, _render_pages_to_jpeg
else:
    from scripts.pdf_route_extract import _ocr_image, _pdf_pages, _render_pages_to_jpeg


def _http_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _summarize_from_text(chat_url: str, model: str, text: str) -> Dict[str, Any]:
    prompt = (
        "Please summarize this PDF page in concise English. "
        "Focus on the key informational points only. "
        "Use 4 short bullet points.\n\n"
        f"Page OCR text:\n{text}"
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "extra_body": {"think": False},
        "max_tokens": 220,
    }
    t0 = time.time()
    body = _http_json(chat_url.rstrip("/") + "/v1/chat/completions", payload)
    return {
        "elapsed_sec": round(time.time() - t0, 3),
        "response": body,
    }


def _summarize_from_image(chat_url: str, model: str, image_path: Path) -> Dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please summarize this PDF page in concise English. "
                            "Focus on the key informational points only. "
                            "Use 4 short bullet points."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "extra_body": {"think": False},
        "max_tokens": 220,
    }
    t0 = time.time()
    body = _http_json(chat_url.rstrip("/") + "/v1/chat/completions", payload)
    return {
        "elapsed_sec": round(time.time() - t0, 3),
        "response": body,
    }


def _message_text(body: Dict[str, Any]) -> str:
    return (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )


def _usage(body: Dict[str, Any]) -> Dict[str, Any]:
    return body.get("usage", {})


def _timings(body: Dict[str, Any]) -> Dict[str, Any]:
    return body.get("timings", {})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare OCR+35B vs direct 35B multimodal on a PDF")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8008")
    parser.add_argument("--chat-model", default="qwen35-35b")
    parser.add_argument("--ocr-url", default="http://127.0.0.1:8012")
    parser.add_argument("--ocr-model", default="ocr-lite")
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages = _pdf_pages(pdf_path)
    end_page = pages if args.last_page is None else min(args.last_page, pages)
    page_results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pdf_compare_") as tmpdir:
        rendered = _render_pages_to_jpeg(
            pdf_path,
            Path(tmpdir),
            first_page=args.first_page,
            last_page=end_page,
            dpi=args.dpi,
        )
        for page_num, image_path in zip(range(args.first_page, end_page + 1), rendered):
            ocr_result = _ocr_image(args.ocr_url, args.ocr_model, image_path)
            ocr_body = ocr_result["response"]
            ocr_text = _message_text(ocr_body)

            ocr35b = _summarize_from_text(args.chat_url, args.chat_model, ocr_text)
            ocr35b_body = ocr35b["response"]

            vlm35b = _summarize_from_image(args.chat_url, args.chat_model, image_path)
            vlm35b_body = vlm35b["response"]

            page_results.append(
                {
                    "page": page_num,
                    "ocr": {
                        "elapsed_sec": ocr_result["elapsed_sec"],
                        "text_chars": len(ocr_text),
                        "usage": _usage(ocr_body),
                        "timings": _timings(ocr_body),
                        "text": ocr_text,
                    },
                    "ocr_plus_35b": {
                        "elapsed_sec": ocr35b["elapsed_sec"],
                        "usage": _usage(ocr35b_body),
                        "timings": _timings(ocr35b_body),
                        "summary": _message_text(ocr35b_body),
                    },
                    "vlm_35b": {
                        "elapsed_sec": vlm35b["elapsed_sec"],
                        "usage": _usage(vlm35b_body),
                        "timings": _timings(vlm35b_body),
                        "summary": _message_text(vlm35b_body),
                    },
                    "end_to_end_sec": {
                        "ocr_plus_35b": round(ocr_result["elapsed_sec"] + ocr35b["elapsed_sec"], 3),
                        "vlm_35b": vlm35b["elapsed_sec"],
                    },
                }
            )

    total_ocr = round(sum(item["ocr"]["elapsed_sec"] for item in page_results), 3)
    total_ocr35b = round(sum(item["ocr_plus_35b"]["elapsed_sec"] for item in page_results), 3)
    total_vlm = round(sum(item["vlm_35b"]["elapsed_sec"] for item in page_results), 3)
    total_e2e_ocr35b = round(sum(item["end_to_end_sec"]["ocr_plus_35b"] for item in page_results), 3)
    total_e2e_vlm = round(sum(item["end_to_end_sec"]["vlm_35b"] for item in page_results), 3)

    result = {
        "pdf": str(pdf_path),
        "first_page": args.first_page,
        "last_page": end_page,
        "page_count": len(page_results),
        "summary": {
            "ocr_only_total_sec": total_ocr,
            "ocr_plus_35b_total_sec": total_e2e_ocr35b,
            "direct_vlm_35b_total_sec": total_e2e_vlm,
            "ocr_plus_35b_avg_sec_per_page": round(total_e2e_ocr35b / len(page_results), 3),
            "direct_vlm_35b_avg_sec_per_page": round(total_e2e_vlm / len(page_results), 3),
            "ocr_stage_total_sec": total_ocr,
            "ocr35b_stage_total_sec": total_ocr35b,
            "vlm35b_stage_total_sec": total_vlm,
        },
        "pages": page_results,
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
