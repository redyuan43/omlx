#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Route PDF extraction between direct text and OCR."""

from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List


def _run_command(args: list[str]) -> str:
    return subprocess.check_output(args, text=True, errors="ignore")


def _pdf_pages(pdf_path: Path) -> int:
    info = _run_command(["pdfinfo", str(pdf_path)])
    match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
    if not match:
        raise RuntimeError(f"unable to determine page count for {pdf_path}")
    return int(match.group(1))


def _pdftotext_sample(pdf_path: Path, *, first_page: int = 1, last_page: int = 2) -> str:
    return _run_command(
        [
            "pdftotext",
            "-f",
            str(first_page),
            "-l",
            str(last_page),
            str(pdf_path),
            "-",
        ]
    )


def _normalize_visible_text(text: str) -> str:
    return "".join(ch for ch in text if ch.isprintable())


def _artifact_score(text: str) -> int:
    patterns = (
        r"\b\w*0o\w*\b",
        r"Mr\.\.",
        r"[A-Za-z]{4,}&[A-Za-z]{4,}",
        r"[A-Za-z]+-of\b",
        r"\b\w*\d+[A-Za-z]\w*\b",
        r"\bEmpioyment\b",
        r"\big7db\b",
        r"\bOUCED\b",
    )
    return sum(len(re.findall(pattern, text)) for pattern in patterns)


def classify_pdf_route(pdf_path: Path) -> Dict[str, Any]:
    sample = _normalize_visible_text(_pdftotext_sample(pdf_path))
    alpha_chars = sum(ch.isalpha() for ch in sample)
    weird_chars = sum(ch in "&^_|~`" for ch in sample)
    artifact_score = _artifact_score(sample)
    text_layer_quality = (
        "good"
        if alpha_chars >= 400 and weird_chars <= 1 and artifact_score <= 10
        else "poor"
    )
    preferred_path = "direct_text" if text_layer_quality == "good" else "ocr"
    return {
        "path": str(pdf_path),
        "pages": _pdf_pages(pdf_path),
        "sample_chars": len(sample),
        "alpha_chars": alpha_chars,
        "weird_chars": weird_chars,
        "artifact_score": artifact_score,
        "text_layer_quality": text_layer_quality,
        "preferred_path": preferred_path,
        "sample_preview": sample[:1000],
    }


def extract_direct_text(pdf_path: Path, *, first_page: int = 1, last_page: int | None = None) -> Dict[str, Any]:
    pages = _pdf_pages(pdf_path)
    end_page = pages if last_page is None else min(last_page, pages)
    t0 = time.time()
    text = _run_command(
        [
            "pdftotext",
            "-f",
            str(first_page),
            "-l",
            str(end_page),
            str(pdf_path),
            "-",
        ]
    )
    return {
        "mode": "direct_text",
        "first_page": first_page,
        "last_page": end_page,
        "elapsed_sec": round(time.time() - t0, 3),
        "text": text,
    }


def _render_pages_to_jpeg(pdf_path: Path, output_dir: Path, *, first_page: int, last_page: int, dpi: int) -> List[Path]:
    prefix = output_dir / "page"
    subprocess.check_call(
        [
            "pdftoppm",
            "-f",
            str(first_page),
            "-l",
            str(last_page),
            "-jpeg",
            "-r",
            str(dpi),
            str(pdf_path),
            str(prefix),
        ]
    )
    return sorted(output_dir.glob("page-*.jpg"))


def _http_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _ocr_image(ocr_url: str, model: str, image_path: Path) -> Dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "stream": False,
        "ocr": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please OCR this image and output the text only. Preserve line breaks where reasonable. Do not summarize.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1200,
    }
    t0 = time.time()
    body = _http_json(ocr_url.rstrip("/") + "/v1/chat/completions", payload)
    elapsed = round(time.time() - t0, 3)
    return {
        "elapsed_sec": elapsed,
        "response": body,
    }


def extract_via_ocr(
    pdf_path: Path,
    *,
    ocr_url: str,
    model: str,
    first_page: int = 1,
    last_page: int | None = None,
    dpi: int = 150,
) -> Dict[str, Any]:
    pages = _pdf_pages(pdf_path)
    end_page = pages if last_page is None else min(last_page, pages)
    with tempfile.TemporaryDirectory(prefix="pdf_ocr_") as tmpdir:
        rendered = _render_pages_to_jpeg(
            pdf_path,
            Path(tmpdir),
            first_page=first_page,
            last_page=end_page,
            dpi=dpi,
        )
        page_results = []
        for offset, image_path in enumerate(rendered, start=first_page):
            ocr_result = _ocr_image(ocr_url, model, image_path)
            body = ocr_result["response"]
            page_results.append(
                {
                    "page": offset,
                    "elapsed_sec": ocr_result["elapsed_sec"],
                    "prompt_tokens": body.get("usage", {}).get("prompt_tokens"),
                    "completion_tokens": body.get("usage", {}).get("completion_tokens"),
                    "prompt_ms": body.get("timings", {}).get("prompt_ms"),
                    "predicted_ms": body.get("timings", {}).get("predicted_ms"),
                    "text": body["choices"][0]["message"]["content"],
                }
            )
    return {
        "mode": "ocr",
        "first_page": first_page,
        "last_page": end_page,
        "dpi": dpi,
        "pages": page_results,
        "text": "\n\n".join(
            f"[page {item['page']}]\n{item['text']}" for item in page_results
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Route PDF extraction between direct text and OCR")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--ocr-url", default="http://127.0.0.1:8012")
    parser.add_argument("--ocr-model", default="ocr-lite")
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--force-mode", choices=("direct_text", "ocr"), default=None)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")
    for binary in ("pdfinfo", "pdftotext", "pdftoppm"):
        if shutil.which(binary) is None:
            raise SystemExit(f"required binary not found: {binary}")

    route = classify_pdf_route(pdf_path)
    mode = args.force_mode or route["preferred_path"]
    if mode == "direct_text":
        extraction = extract_direct_text(
            pdf_path,
            first_page=args.first_page,
            last_page=args.last_page,
        )
    else:
        extraction = extract_via_ocr(
            pdf_path,
            ocr_url=args.ocr_url,
            model=args.ocr_model,
            first_page=args.first_page,
            last_page=args.last_page,
            dpi=args.dpi,
        )
    payload = {
        "route": route,
        "extraction": extraction,
    }
    output = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
