#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Smoke-test multimodal routing and capability reporting through the DGX control plane."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict

_PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y2Rgx8AAAAASUVORK5CYII="
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
                {"type": "image_url", "image_url": {"url": _PNG_DATA_URL}},
            ],
        }
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multimodal routing smoke benchmark")
    parser.add_argument("--control-plane-url", default="http://127.0.0.1:8008")
    parser.add_argument("--model", default="qwen35-4b")
    args = parser.parse_args()

    started = time.perf_counter()
    health = _http_json(f"{args.control_plane_url}/health")
    capabilities = _http_json(f"{args.control_plane_url}/admin/api/runtime/capabilities")
    models = _http_json(f"{args.control_plane_url}/v1/models")

    vision_payload = {
        "model": args.model,
        "messages": _image_messages("Describe this image in one short sentence."),
        "temperature": 0,
        "max_tokens": 32,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    vision_status, vision_response = _http_json_allow_error(
        f"{args.control_plane_url}/v1/chat/completions",
        method="POST",
        payload=vision_payload,
    )

    ocr_payload = {
        "model": args.model,
        "messages": _image_messages("Read the text in this image."),
        "temperature": 0,
        "max_tokens": 64,
        "chat_template_kwargs": {"enable_thinking": False},
        "metadata": {"omlx_task": "ocr"},
    }
    ocr_status, ocr_response = _http_json_allow_error(
        f"{args.control_plane_url}/v1/chat/completions",
        method="POST",
        payload=ocr_payload,
    )

    duration = round(time.perf_counter() - started, 3)
    print(
        json.dumps(
            {
                "urls": {"control_plane_url": args.control_plane_url},
                "health": health,
                "capabilities": capabilities,
                "models": models,
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
                    "vision_supported": capabilities.get("backend", {})
                    .get("capabilities", {})
                    .get("vision_chat"),
                    "ocr_supported": capabilities.get("backend", {})
                    .get("capabilities", {})
                    .get("ocr"),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
