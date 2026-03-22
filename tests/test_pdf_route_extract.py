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
