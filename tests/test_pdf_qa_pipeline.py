# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[1]
    script_path = root / "scripts" / "pdf_qa_pipeline.py"
    spec = importlib.util.spec_from_file_location("pdf_qa_pipeline", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pdf_qa_pipeline = _load_module()


def test_http_json_posts_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(pdf_qa_pipeline.urllib.request, "urlopen", fake_urlopen)

    result = pdf_qa_pipeline._http_json(
        "http://127.0.0.1:8008/v1/chat/completions",
        {"x": 1},
    )

    assert result == {"ok": True}
    assert captured["url"] == "http://127.0.0.1:8008/v1/chat/completions"
    assert captured["data"] == b'{"x": 1}'


def test_chunk_text_splits_pages_and_limits_size():
    chunks = pdf_qa_pipeline._chunk_text(
        "Alpha\n\nBeta\n\n\f\nGamma\n\nDelta",
        max_chars=8,
    )

    assert [chunk["page"] for chunk in chunks] == [1, 1, 2, 2]
    assert chunks[0]["text"] == "Alpha"
    assert chunks[1]["text"] == "Beta"
    assert chunks[2]["text"] == "Gamma"


def test_chunk_extraction_keeps_body_and_image_caption_separate():
    extraction = {
        "mode": "page_auto",
        "pages": [
            {
                "page": 3,
                "route": "ocr_text_plus_vlm_caption",
                "body_text": "Main body text here.",
                "image_caption": "Chart shows growth over time.",
            }
        ],
    }

    chunks = pdf_qa_pipeline._chunk_extraction(extraction, max_chars=1200)

    assert len(chunks) == 2
    assert chunks[0]["page"] == 3
    assert chunks[0]["kind"] == "body_text"
    assert chunks[0]["text"] == "Main body text here."
    assert chunks[1]["page"] == 3
    assert chunks[1]["kind"] == "image_caption"
    assert chunks[1]["text"] == "Chart shows growth over time."
