# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "pdf_route_extract.py"
    spec = importlib.util.spec_from_file_location("pdf_route_extract", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pdf_route_extract = _load_module()


def test_classify_pdf_route_prefers_direct_text_for_clean_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(pdf_route_extract, "_pdf_pages", lambda path: 12)
    monkeypatch.setattr(
        pdf_route_extract,
        "_pdftotext_sample",
        lambda path, first_page=1, last_page=2: "A" * 600,
    )

    result = pdf_route_extract.classify_pdf_route(pdf_path)

    assert result["pages"] == 12
    assert result["text_layer_quality"] == "good"
    assert result["preferred_path"] == "direct_text"


def test_classify_pdf_route_prefers_ocr_for_noisy_text(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(pdf_route_extract, "_pdf_pages", lambda path: 29)
    monkeypatch.setattr(
        pdf_route_extract,
        "_pdftotext_sample",
        lambda path, first_page=1, last_page=2: (
            "DOCUMENT RESUME Empioyment Oppor&nities Mr..Herbeit "
            "INSTITUTIONBureau Bureau-of 42,ESCRIPTORS ig7db Market"
        ),
    )

    result = pdf_route_extract.classify_pdf_route(pdf_path)

    assert result["pages"] == 29
    assert result["artifact_score"] > 4
    assert result["text_layer_quality"] == "poor"
    assert result["preferred_path"] == "ocr"


def test_extract_with_page_strategy_uses_direct_text_without_vlm(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("stub", encoding="utf-8")

    monkeypatch.setattr(pdf_route_extract, "_pdf_pages", lambda path: 1)
    monkeypatch.setattr(
        pdf_route_extract,
        "_pdftotext_page",
        lambda path, page: "A" * 600,
    )
    monkeypatch.setattr(
        pdf_route_extract,
        "_render_page_to_jpeg",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not render direct text page")),
    )

    result = pdf_route_extract.extract_with_page_strategy(
        pdf_path,
        chat_url="http://127.0.0.1:8008",
        chat_model="qwen35-35b",
        ocr_url="http://127.0.0.1:8012",
        ocr_model="ocr-lite",
    )

    assert result["document_summary"]["direct_text_pages"] == 1
    assert result["document_summary"]["vlm_invoked_pages"] == 0
    assert result["pages"][0]["route"] == "direct_text"
    assert result["pages"][0]["body_text"] == "A" * 600
    assert result["pages"][0]["image_caption"] == ""


def test_extract_with_page_strategy_routes_sparse_page_to_ocr_plus_vlm(monkeypatch, tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("stub", encoding="utf-8")
    image_path = tmp_path / "page-1.jpg"
    image_path.write_bytes(b"jpg")

    monkeypatch.setattr(pdf_route_extract, "_pdf_pages", lambda path: 1)
    monkeypatch.setattr(
        pdf_route_extract,
        "_pdftotext_page",
        lambda path, page: "Fig. 1",
    )
    monkeypatch.setattr(
        pdf_route_extract,
        "_render_page_to_jpeg",
        lambda *args, **kwargs: image_path,
    )
    monkeypatch.setattr(
        pdf_route_extract,
        "_vlm_page_review",
        lambda *args, **kwargs: {
            "elapsed_sec": 1.2,
            "response": {"usage": {}, "timings": {}},
            "raw_content": '{"route":"ocr_text_plus_vlm_caption"}',
            "decision": {
                "route": "ocr_text_plus_vlm_caption",
                "page_kind": "mixed",
                "confidence": "high",
                "reason": "page contains chart and short labels",
                "image_caption": "Bar chart comparing categories over time.",
            },
        },
    )
    monkeypatch.setattr(
        pdf_route_extract,
        "_ocr_image",
        lambda *args, **kwargs: {
            "elapsed_sec": 2.3,
            "response": {
                "usage": {},
                "timings": {},
                "choices": [{"message": {"content": "Legend and axis labels"}}],
            },
        },
    )

    result = pdf_route_extract.extract_with_page_strategy(
        pdf_path,
        chat_url="http://127.0.0.1:8008",
        chat_model="qwen35-35b",
        ocr_url="http://127.0.0.1:8012",
        ocr_model="ocr-lite",
    )

    assert result["document_summary"]["ocr_text_plus_vlm_caption_pages"] == 1
    assert result["document_summary"]["vlm_invoked_pages"] == 1
    assert result["pages"][0]["route"] == "ocr_text_plus_vlm_caption"
    assert result["pages"][0]["body_text"] == "Legend and axis labels"
    assert result["pages"][0]["image_caption"] == "Bar chart comparing categories over time."
