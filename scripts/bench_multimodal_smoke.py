#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test embeddings, rerank, VLM, and OCR routing through the DGX control plane."""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict

from PIL import Image, ImageDraw, ImageFont

_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAoAAAADwCAIAAAAfEkKcAAAeRUlEQVR4nO3deVyU5f7/8QGURUUUApfcFzwmggpIqYR1LEQljzsq9cgULVNz63TcjlqdtMU8melx6Zh7iuaC+76k4QilRCqLeFQwTASRZVBmmO8f/PLhz2Xua9ZrgNfzT73muj9zD495z31f93VdDnq9XgUAAGzLUXYBAABURQQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhAAAMAIAEBDACABAQwAAASEMAAAEhQTXYBRvvjjz/UarVarU5JSbl27VpmZmZhYWFxcbFOp6tRo4abm5uPj0+TJk2aN28eGBgYHBzcrl07BwcHGxd569ats2fPJiQkXLp06fr165mZmXfv3tVoNPfv33d3d/fw8PDw8KhXr56/v0+H6dixY4cOHd577z0R4eHhJSUln376qcrJyVqlUtGxY0dJScnq1avT09Nbt26tra2NiIgQ2bm7u69evbqkpERwO3ToULNmzSQlJbP2lq6uruLFiyspKSk4ODh27NjU1FT+9+TJk4qLi5OTk0W3d0JCQhgxYkT//v2pVKrKysqQkBCXL1/eu3evpaXlyJEjV65c2bx5s9jYWEVFhcltDBw4cPXq1fPmzRPS3RMrKysrK6t2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"
)


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_json_allow_error(
    url: str,
    *,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
) -> tuple[int, Dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        return exc.code, parsed


def _image_messages(prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _ocr_test_image_data_url()}},
            ],
        }
    ]


def _ocr_test_image_data_url() -> str:
    image = Image.new("RGB", (640, 240), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            64,
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 80), "OCR TEST 123", fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _pick_model(
    models_payload: Dict[str, Any],
    capability_name: str,
    explicit: str,
) -> str:
    if explicit:
        return explicit
    for item in models_payload.get("data", []):
        if not isinstance(item, dict):
            continue
        capabilities = item.get("capabilities") or {}
        if capabilities.get(capability_name):
            model_id = item.get("id")
            if model_id:
                return str(model_id)
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal routing smoke benchmark")
    parser.add_argument("--control-plane-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", default="")
    parser.add_argument("--vision-model", default="")
    parser.add_argument("--ocr-model", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--rerank-model", default="")
    args = parser.parse_args()

    started = time.perf_counter()
    health = _http_json(f"{args.control_plane_url}/health")
    capabilities = _http_json(f"{args.control_plane_url}/admin/api/runtime/capabilities")
    models = _http_json(f"{args.control_plane_url}/v1/models")
    vision_model = _pick_model(models, "vision_chat", args.vision_model or args.model)
    ocr_model = _pick_model(models, "ocr", args.ocr_model)
    embedding_model = _pick_model(models, "embeddings", args.embedding_model)
    rerank_model = _pick_model(models, "rerank", args.rerank_model)

    vision_payload = {
        "model": vision_model,
        "messages": _image_messages("Describe this image in one short sentence."),
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if vision_model:
        vision_status, vision_response = _http_json_allow_error(
            f"{args.control_plane_url}/v1/chat/completions",
            method="POST",
            payload=vision_payload,
        )
    else:
        vision_status, vision_response = 0, {"detail": "no vision model available"}

    ocr_payload = {
        "model": ocr_model,
        "messages": _image_messages("Read the text in this image."),
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "metadata": {"omlx_task": "ocr"},
    }
    if ocr_model:
        ocr_status, ocr_response = _http_json_allow_error(
            f"{args.control_plane_url}/v1/chat/completions",
            method="POST",
            payload=ocr_payload,
        )
    else:
        ocr_status, ocr_response = 0, {"detail": "no ocr model available"}

    embeddings_payload = {
        "model": embedding_model,
        "input": "hello embeddings",
    }
    if embedding_model:
        embeddings_status, embeddings_response = _http_json_allow_error(
            f"{args.control_plane_url}/v1/embeddings",
            method="POST",
            payload=embeddings_payload,
        )
    else:
        embeddings_status, embeddings_response = 0, {"detail": "no embedding model available"}

    rerank_payload = {
        "model": rerank_model,
        "query": "apple fruit",
        "documents": [
            "fresh apple fruit",
            "gpu benchmark report",
            "banana smoothie recipe",
        ],
    }
    if rerank_model:
        rerank_status, rerank_response = _http_json_allow_error(
            f"{args.control_plane_url}/v1/rerank",
            method="POST",
            payload=rerank_payload,
        )
    else:
        rerank_status, rerank_response = 0, {"detail": "no rerank model available"}

    duration = round(time.perf_counter() - started, 3)
    print(
        json.dumps(
            {
                "urls": {"control_plane_url": args.control_plane_url},
                "health": health,
                "capabilities": capabilities,
                "models": models,
                "selected_models": {
                    "vision_model": vision_model,
                    "ocr_model": ocr_model,
                    "embedding_model": embedding_model,
                    "rerank_model": rerank_model,
                },
                "embeddings_request": {
                    "status_code": embeddings_status,
                    "ok": 200 <= embeddings_status < 300,
                    "response": embeddings_response,
                },
                "rerank_request": {
                    "status_code": rerank_status,
                    "ok": 200 <= rerank_status < 300,
                    "response": rerank_response,
                },
                "vision_request": {
                    "status_code": vision_status,
                    "ok": 200 <= vision_status < 300,
                    "response": vision_response,
                },
                "ocr_request": {
                    "status_code": ocr_status,
                    "ok": 200 <= ocr_status < 300,
                    "response": ocr_response,
                },
                "summary": {
                    "duration_sec": duration,
                    "backend_kind": capabilities.get("backend", {}).get("kind"),
                    "embedding_model": embedding_model,
                    "rerank_model": rerank_model,
                    "vision_model": vision_model,
                    "ocr_model": ocr_model,
                    "embeddings_supported": capabilities.get("backend", {})
                    .get("capabilities", {})
                    .get("embeddings"),
                    "rerank_supported": capabilities.get("backend", {})
                    .get("capabilities", {})
                    .get("rerank"),
                    "vision_supported": capabilities.get("backend", {})
                    .get("capabilities", {})
                    .get("vision_chat"),
                    "ocr_supported": capabilities.get("backend", {})
                    .get("capabilities", {})
                    .get("ocr"),
                    "embeddings_ok": 200 <= embeddings_status < 300,
                    "rerank_ok": 200 <= rerank_status < 300,
                    "vision_ok": 200 <= vision_status < 300,
                    "ocr_ok": 200 <= ocr_status < 300,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
