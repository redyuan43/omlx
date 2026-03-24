#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Route PDF extraction between direct text, OCR, and conditional VLM review."""

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


def _pdftotext_page(pdf_path: Path, page: int) -> str:
    return _run_command(
        [
            "pdftotext",
            "-f",
            str(page),
            "-l",
            str(page),
            str(pdf_path),
            "-",
        ]
    )


def _normalize_visible_text(text: str) -> str:
    return "".join(ch for ch in text if ch.isprintable() or ch in "\n\t")


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


def _count_lines_with(pattern: str, text: str) -> int:
    return sum(1 for line in text.splitlines() if re.search(pattern, line))


def _page_text_signals(text: str) -> Dict[str, Any]:
    sample = _normalize_visible_text(text)
    alpha_chars = sum(ch.isalpha() for ch in sample)
    digit_chars = sum(ch.isdigit() for ch in sample)
    weird_chars = sum(ch in "&^_|~`" for ch in sample)
    artifact_score = _artifact_score(sample)
    text_chars = len(sample.strip())
    line_count = len([line for line in sample.splitlines() if line.strip()])
    numeric_lines = _count_lines_with(r"\d", sample)
    table_like_lines = _count_lines_with(r"(\|)|(\s{3,})|(\b\d[\d,.-]*\b.*\b\d[\d,.-]*\b)", sample)

    if text_chars < 80 or alpha_chars < 50:
        text_layer_quality = "missing"
    elif alpha_chars >= 400 and weird_chars <= 1 and artifact_score <= 10:
        text_layer_quality = "good"
    else:
        text_layer_quality = "poor"

    likely_table = table_like_lines >= 3 and numeric_lines >= 2
    return {
        "sample_preview": sample[:1000],
        "text_chars": text_chars,
        "alpha_chars": alpha_chars,
        "digit_chars": digit_chars,
        "weird_chars": weird_chars,
        "artifact_score": artifact_score,
        "line_count": line_count,
        "numeric_lines": numeric_lines,
        "table_like_lines": table_like_lines,
        "likely_table": likely_table,
        "text_layer_quality": text_layer_quality,
    }


def classify_pdf_route(pdf_path: Path) -> Dict[str, Any]:
    sample = _normalize_visible_text(_pdftotext_sample(pdf_path))
    signals = _page_text_signals(sample)
    text_layer_quality = signals["text_layer_quality"]
    preferred_path = "direct_text" if text_layer_quality == "good" else "ocr"
    return {
        "path": str(pdf_path),
        "pages": _pdf_pages(pdf_path),
        "sample_chars": len(sample),
        "alpha_chars": signals["alpha_chars"],
        "weird_chars": signals["weird_chars"],
        "artifact_score": signals["artifact_score"],
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


def _render_page_to_jpeg(pdf_path: Path, output_dir: Path, *, page: int, dpi: int) -> Path:
    page_dir = output_dir / f"page_{page}"
    page_dir.mkdir(parents=True, exist_ok=True)
    rendered = _render_pages_to_jpeg(
        pdf_path,
        page_dir,
        first_page=page,
        last_page=page,
        dpi=dpi,
    )
    if not rendered:
        raise RuntimeError(f"failed to render page {page} for {pdf_path}")
    return rendered[0]


def _http_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        return json.loads(match.group(0))


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


def _vlm_page_review(
    chat_url: str,
    model: str,
    image_path: Path,
    *,
    page_num: int,
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    prompt = (
        "You are routing a PDF page for extraction and RAG.\n"
        "Decide which route is best for this page.\n"
        "Allowed routes: direct_text, ocr_text, ocr_text_plus_vlm_caption, vlm_only.\n"
        "Prefer faithful text extraction for readable text and tables. "
        "Use ocr_text_plus_vlm_caption when the page has meaningful charts, figures, diagrams, or images "
        "that need explanation in addition to extracted text. Use vlm_only only when the page is mostly visual "
        "and reliable text extraction is not the main value.\n"
        f"Page number: {page_num}\n"
        f"Text layer quality hint: {signals['text_layer_quality']}\n"
        f"Likely table: {str(signals['likely_table']).lower()}\n"
        f"Text chars hint: {signals['text_chars']}\n"
        f"Artifact score hint: {signals['artifact_score']}\n"
        f"Preview of current text layer extraction:\n{signals['sample_preview'][:800]}\n\n"
        "Return JSON only with keys: "
        "route, page_kind, confidence, reason, image_caption.\n"
        "page_kind must be one of: text_dense, mixed, visual.\n"
        "image_caption should be an empty string unless the page contains useful visual information that should "
        "be indexed separately for RAG."
    )
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }
        ],
        "extra_body": {"think": False},
        "temperature": 0,
        "max_tokens": 260,
    }
    t0 = time.time()
    body = _http_json(chat_url.rstrip("/") + "/v1/chat/completions", payload)
    content = (
        body.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
        .strip()
    )
    parsed = _extract_json_object(content)
    return {
        "elapsed_sec": round(time.time() - t0, 3),
        "response": body,
        "decision": parsed,
        "raw_content": content,
    }


def _default_page_route(signals: Dict[str, Any]) -> Dict[str, Any]:
    quality = signals["text_layer_quality"]
    numeric_heavy = signals["digit_chars"] >= 40 and signals["line_count"] >= 8
    visual_candidate = signals["likely_table"] or numeric_heavy
    if visual_candidate and quality != "missing":
        return {
            "route": "pending_vlm",
            "page_kind": "mixed",
            "confidence": "medium",
            "reason": "table-like or numeric-heavy page needs visual review",
            "image_caption": "",
            "vlm_review": True,
        }
    if quality == "good" and signals["text_chars"] >= 250:
        return {
            "route": "direct_text",
            "page_kind": "text_dense",
            "confidence": "high",
            "reason": "clean text layer with enough text",
            "image_caption": "",
            "vlm_review": False,
        }
    if quality == "poor" and signals["text_chars"] >= 250:
        return {
            "route": "ocr_text",
            "page_kind": "text_dense" if signals["likely_table"] else "mixed",
            "confidence": "high",
            "reason": "text layer is noisy but the page is text-heavy",
            "image_caption": "",
            "vlm_review": False,
        }
    return {
        "route": "pending_vlm",
        "page_kind": "mixed",
        "confidence": "medium",
        "reason": "sparse or ambiguous page needs visual review",
        "image_caption": "",
        "vlm_review": True,
    }


def _resolve_vlm_route(signals: Dict[str, Any], review: Dict[str, Any]) -> Dict[str, Any]:
    decision = review.get("decision") or {}
    route = str(decision.get("route") or "").strip()
    if route not in {"direct_text", "ocr_text", "ocr_text_plus_vlm_caption", "vlm_only"}:
        route = "ocr_text_plus_vlm_caption" if signals["text_chars"] >= 120 else "vlm_only"
    if route == "direct_text" and signals["text_layer_quality"] != "good":
        route = "ocr_text"
    if route == "vlm_only" and signals["text_chars"] >= 250:
        route = "ocr_text_plus_vlm_caption"
    return {
        "route": route,
        "page_kind": str(decision.get("page_kind") or "mixed"),
        "confidence": str(decision.get("confidence") or "medium"),
        "reason": str(decision.get("reason") or "vlm review"),
        "image_caption": str(decision.get("image_caption") or "").strip(),
        "vlm_review": True,
    }


def _build_page_record(
    *,
    page: int,
    route_decision: Dict[str, Any],
    signals: Dict[str, Any],
    body_text: str,
    image_caption: str,
    source_model: Dict[str, str],
    text_extract_elapsed_sec: float,
    vlm_result: Dict[str, Any] | None = None,
    ocr_result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    ocr_elapsed = 0.0 if ocr_result is None else float(ocr_result["elapsed_sec"])
    vlm_elapsed = 0.0 if vlm_result is None else float(vlm_result["elapsed_sec"])
    record = {
        "page": page,
        "route": route_decision["route"],
        "route_reason": route_decision["reason"],
        "page_kind": route_decision["page_kind"],
        "confidence": route_decision["confidence"],
        "text_layer_quality": signals["text_layer_quality"],
        "vlm_invoked": bool(route_decision.get("vlm_review")),
        "body_text": body_text,
        "image_caption": image_caption,
        "source_model": source_model,
        "elapsed_sec": round(text_extract_elapsed_sec + ocr_elapsed + vlm_elapsed, 3),
        "stage_timings": {
            "text_extract_sec": round(text_extract_elapsed_sec, 3),
            "ocr_sec": round(ocr_elapsed, 3),
            "vlm_sec": round(vlm_elapsed, 3),
        },
        "signals": {
            "text_chars": signals["text_chars"],
            "alpha_chars": signals["alpha_chars"],
            "digit_chars": signals["digit_chars"],
            "weird_chars": signals["weird_chars"],
            "artifact_score": signals["artifact_score"],
            "likely_table": signals["likely_table"],
        },
    }
    if ocr_result is not None:
        body = ocr_result["response"]
        record["ocr"] = {
            "elapsed_sec": ocr_result["elapsed_sec"],
            "usage": body.get("usage", {}),
            "timings": body.get("timings", {}),
        }
    if vlm_result is not None:
        body = vlm_result["response"]
        record["vlm"] = {
            "elapsed_sec": vlm_result["elapsed_sec"],
            "usage": body.get("usage", {}),
            "timings": body.get("timings", {}),
            "raw_content": vlm_result.get("raw_content", ""),
        }
    return record


def extract_with_page_strategy(
    pdf_path: Path,
    *,
    chat_url: str,
    chat_model: str,
    ocr_url: str,
    ocr_model: str,
    first_page: int = 1,
    last_page: int | None = None,
    dpi: int = 150,
    force_mode: str | None = None,
) -> Dict[str, Any]:
    pages = _pdf_pages(pdf_path)
    end_page = pages if last_page is None else min(last_page, pages)
    route_counts = {
        "direct_text": 0,
        "ocr_text": 0,
        "ocr_text_plus_vlm_caption": 0,
        "vlm_only": 0,
    }
    page_results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="pdf_route_auto_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for page_num in range(first_page, end_page + 1):
            text_t0 = time.time()
            page_text = _pdftotext_page(pdf_path, page_num)
            text_extract_elapsed_sec = round(time.time() - text_t0, 3)
            signals = _page_text_signals(page_text)
            review = None
            image_path: Path | None = None

            if force_mode == "direct_text":
                decision = {
                    "route": "direct_text",
                    "page_kind": "text_dense",
                    "confidence": "forced",
                    "reason": "forced direct_text",
                    "image_caption": "",
                    "vlm_review": False,
                }
            elif force_mode == "ocr":
                decision = {
                    "route": "ocr_text",
                    "page_kind": "text_dense",
                    "confidence": "forced",
                    "reason": "forced ocr",
                    "image_caption": "",
                    "vlm_review": False,
                }
            else:
                decision = _default_page_route(signals)
                if decision["route"] == "pending_vlm":
                    image_path = _render_page_to_jpeg(
                        pdf_path,
                        tmpdir_path,
                        page=page_num,
                        dpi=dpi,
                    )
                    review = _vlm_page_review(
                        chat_url,
                        chat_model,
                        image_path,
                        page_num=page_num,
                        signals=signals,
                    )
                    decision = _resolve_vlm_route(signals, review)
                else:
                    review = None

            body_text = ""
            image_caption = ""
            source_model = {"body_text": "", "image_caption": ""}
            ocr_result = None
            vlm_result = review

            if decision["route"] == "direct_text":
                body_text = page_text.strip()
                source_model["body_text"] = "pdftotext"
            elif decision["route"] == "ocr_text":
                if image_path is None:
                    image_path = _render_page_to_jpeg(
                        pdf_path,
                        tmpdir_path,
                        page=page_num,
                        dpi=dpi,
                    )
                ocr_result = _ocr_image(ocr_url, ocr_model, image_path)
                body_text = (
                    ocr_result["response"].get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                )
                source_model["body_text"] = ocr_model
            elif decision["route"] == "ocr_text_plus_vlm_caption":
                if image_path is None:
                    image_path = _render_page_to_jpeg(
                        pdf_path,
                        tmpdir_path,
                        page=page_num,
                        dpi=dpi,
                    )
                ocr_result = _ocr_image(ocr_url, ocr_model, image_path)
                body_text = (
                    ocr_result["response"].get("choices", [{}])[0].get("message", {}).get("content", "").strip()
                )
                source_model["body_text"] = ocr_model
                image_caption = decision["image_caption"]
                source_model["image_caption"] = chat_model
            elif decision["route"] == "vlm_only":
                image_caption = decision["image_caption"]
                source_model["image_caption"] = chat_model

            route_counts[decision["route"]] += 1
            page_results.append(
                _build_page_record(
                    page=page_num,
                    route_decision=decision,
                    signals=signals,
                    body_text=body_text,
                    image_caption=image_caption,
                    source_model=source_model,
                    text_extract_elapsed_sec=text_extract_elapsed_sec,
                    vlm_result=vlm_result,
                    ocr_result=ocr_result,
                )
            )

    body_text = "\n\n".join(
        f"[page {item['page']}]\n{item['body_text']}"
        for item in page_results
        if item["body_text"].strip()
    )
    image_captions = "\n\n".join(
        f"[page {item['page']}]\n{item['image_caption']}"
        for item in page_results
        if item["image_caption"].strip()
    )
    return {
        "mode": "page_auto",
        "strategy": "extraction_rag",
        "first_page": first_page,
        "last_page": end_page,
        "dpi": dpi,
        "pages": page_results,
        "route_counts": route_counts,
        "text": body_text,
        "image_captions": image_captions,
        "document_summary": {
            "page_count": len(page_results),
            "direct_text_pages": route_counts["direct_text"],
            "ocr_text_pages": route_counts["ocr_text"],
            "ocr_text_plus_vlm_caption_pages": route_counts["ocr_text_plus_vlm_caption"],
            "vlm_only_pages": route_counts["vlm_only"],
            "vlm_invoked_pages": sum(1 for item in page_results if item["vlm_invoked"]),
        },
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
    parser.add_argument("--strategy", default="extraction_rag")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8008")
    parser.add_argument("--chat-model", default="qwen35-35b")
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
    if args.strategy == "extraction_rag":
        extraction = extract_with_page_strategy(
            pdf_path,
            chat_url=args.chat_url,
            chat_model=args.chat_model,
            ocr_url=args.ocr_url,
            ocr_model=args.ocr_model,
            first_page=args.first_page,
            last_page=args.last_page,
            dpi=args.dpi,
            force_mode=args.force_mode,
        )
    else:
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
