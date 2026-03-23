# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "bench_pdf_route_strategy.py"
    spec = importlib.util.spec_from_file_location("bench_pdf_route_strategy", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench_pdf_route_strategy = _load_module()


def test_build_report_computes_route_stats_and_hit_rate():
    extraction = {
        "mode": "page_auto",
        "strategy": "extraction_rag",
        "route_counts": {
            "direct_text": 1,
            "ocr_text": 0,
            "ocr_text_plus_vlm_caption": 1,
            "vlm_only": 0,
        },
        "document_summary": {"vlm_invoked_pages": 1},
        "pages": [
            {
                "page": 1,
                "route": "direct_text",
                "elapsed_sec": 0.015,
                "stage_timings": {"text_extract_sec": 0.015, "ocr_sec": 0.0, "vlm_sec": 0.0},
                "vlm_invoked": False,
                "route_reason": "clean text layer",
            },
            {
                "page": 2,
                "route": "ocr_text_plus_vlm_caption",
                "elapsed_sec": 11.2,
                "stage_timings": {"text_extract_sec": 0.01, "ocr_sec": 4.1, "vlm_sec": 7.09},
                "vlm_invoked": True,
                "route_reason": "chart page",
            },
        ],
    }

    report = bench_pdf_route_strategy._build_report(
        extraction,
        wall_time_sec=11.5,
        expectations={1: "direct_text", 2: "ocr_text_plus_vlm_caption"},
    )

    assert report["page_count"] == 2
    assert report["route_stats"]["direct_text"]["pages"] == 1
    assert report["route_stats"]["ocr_text_plus_vlm_caption"]["vlm_invoked_pages"] == 1
    assert report["expectations"]["overall_hit_rate"] == 1.0
    assert report["expectations"]["route_hit_rate"]["direct_text"]["matched_pages"] == 1
