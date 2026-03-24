#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark document-level PDF routing outcomes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pdf_route_extract import extract_with_page_strategy
else:
    from scripts.pdf_route_extract import extract_with_page_strategy


def _load_expectations(path: Path | None) -> Dict[int, str]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "expected_routes" in raw:
        raw = raw["expected_routes"]
    if not isinstance(raw, dict):
        raise SystemExit("expectation file must be a JSON object keyed by page number")
    return {int(page): str(route) for page, route in raw.items()}


def _build_report(
    extraction: Dict[str, Any],
    *,
    wall_time_sec: float,
    expectations: Dict[int, str],
) -> Dict[str, Any]:
    route_stats: Dict[str, Dict[str, Any]] = {}
    mismatches = []
    matched = 0
    labeled_pages = 0

    for page in extraction.get("pages", []):
        route = page["route"]
        stats = route_stats.setdefault(
            route,
            {
                "pages": 0,
                "total_elapsed_sec": 0.0,
                "avg_elapsed_sec": 0.0,
                "vlm_invoked_pages": 0,
            },
        )
        stats["pages"] += 1
        stats["total_elapsed_sec"] += float(page.get("elapsed_sec", 0.0))
        if page.get("vlm_invoked"):
            stats["vlm_invoked_pages"] += 1

        expected_route = expectations.get(page["page"])
        if expected_route is not None:
            labeled_pages += 1
            is_match = expected_route == route
            if is_match:
                matched += 1
            else:
                mismatches.append(
                    {
                        "page": page["page"],
                        "expected_route": expected_route,
                        "actual_route": route,
                        "route_reason": page.get("route_reason", ""),
                    }
                )

    for stats in route_stats.values():
        if stats["pages"] > 0:
            stats["total_elapsed_sec"] = round(stats["total_elapsed_sec"], 3)
            stats["avg_elapsed_sec"] = round(stats["total_elapsed_sec"] / stats["pages"], 3)

    route_hit_rate: Dict[str, Dict[str, Any]] = {}
    if expectations:
        expected_routes = sorted(set(expectations.values()))
        for route in expected_routes:
            expected_pages = [page for page, expected in expectations.items() if expected == route]
            hits = sum(
                1
                for page in extraction.get("pages", [])
                if page["page"] in expected_pages and page["route"] == route
            )
            route_hit_rate[route] = {
                "expected_pages": len(expected_pages),
                "matched_pages": hits,
                "hit_rate": round(hits / len(expected_pages), 3) if expected_pages else 0.0,
            }

    return {
        "mode": extraction.get("mode"),
        "strategy": extraction.get("strategy"),
        "page_count": len(extraction.get("pages", [])),
        "wall_time_sec": round(wall_time_sec, 3),
        "route_counts": extraction.get("route_counts", {}),
        "route_stats": route_stats,
        "vlm_invoked_pages": extraction.get("document_summary", {}).get("vlm_invoked_pages", 0),
        "expectations": {
            "labeled_pages": labeled_pages,
            "matched_pages": matched,
            "overall_hit_rate": round(matched / labeled_pages, 3) if labeled_pages else None,
            "route_hit_rate": route_hit_rate,
            "mismatches": mismatches,
        },
        "pages": [
            {
                "page": page["page"],
                "route": page["route"],
                "expected_route": expectations.get(page["page"]),
                "matched": None if page["page"] not in expectations else expectations[page["page"]] == page["route"],
                "elapsed_sec": page.get("elapsed_sec", 0.0),
                "stage_timings": page.get("stage_timings", {}),
                "vlm_invoked": page.get("vlm_invoked", False),
            }
            for page in extraction.get("pages", [])
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark PDF routing strategy")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--chat-url", default="http://127.0.0.1:8008")
    parser.add_argument("--chat-model", default="qwen35-35b")
    parser.add_argument("--ocr-url", default="http://127.0.0.1:8012")
    parser.add_argument("--ocr-model", default="ocr-lite")
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--expectations", default=None, help="Optional JSON file of expected routes keyed by page")
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    expectations_path = None if args.expectations is None else Path(args.expectations).expanduser().resolve()
    expectations = _load_expectations(expectations_path)

    t0 = time.time()
    extraction = extract_with_page_strategy(
        pdf_path,
        chat_url=args.chat_url,
        chat_model=args.chat_model,
        ocr_url=args.ocr_url,
        ocr_model=args.ocr_model,
        first_page=args.first_page,
        last_page=args.last_page,
        dpi=args.dpi,
    )
    report = {
        "pdf": str(pdf_path),
        "benchmark": _build_report(
            extraction,
            wall_time_sec=time.time() - t0,
            expectations=expectations,
        ),
    }
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
