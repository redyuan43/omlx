#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a strict AGX parity smoke test against the AMD control planes."""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


def _http_json(
    url: str,
    *,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _http_json_allow_error(
    url: str,
    *,
    method: str = "GET",
    payload: Dict[str, Any] | None = None,
    timeout: float = 120.0,
) -> Tuple[int, Dict[str, Any]]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        parsed = json.loads(body) if body else {}
        return exc.code, parsed


def _assistant_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    choice = choices[0]
    if not isinstance(choice, dict):
        return ""
    message = choice.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    return "" if content is None else str(content)


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


def _test_image_data_url() -> str:
    image = Image.new("RGB", (768, 256), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            64,
        )
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 80), "AGX 395", fill="black", font=font)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _contains_all(text: str, fragments: List[str]) -> bool:
    lowered = text.lower()
    return all(fragment.lower() in lowered for fragment in fragments)


def _nonempty_assistant_text(payload: Dict[str, Any]) -> bool:
    return bool(_assistant_text(payload).strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict AGX parity smoke test for AMD 395")
    parser.add_argument("--main-url", default="http://127.0.0.1:8008")
    parser.add_argument("--ocr-url", default="http://127.0.0.1:8012")
    parser.add_argument("--main-model", default="qwen35-35b")
    parser.add_argument("--ocr-model", default="ocr-lite")
    parser.add_argument("--embedding-model", default="embed-text")
    parser.add_argument("--rerank-model", default="rerank-qwen")
    parser.add_argument("--context-id", default="agx-parity-smoke")
    args = parser.parse_args()

    started = time.perf_counter()
    image_url = _test_image_data_url()

    main_models = _http_json(f"{args.main_url}/v1/models")
    ocr_models = _http_json(f"{args.ocr_url}/v1/models")
    main_runtime = _http_json(f"{args.main_url}/admin/api/runtime")
    main_model = _pick_model(main_models, "vision_chat", args.main_model)
    ocr_model = _pick_model(ocr_models, "ocr", args.ocr_model)
    embedding_model = _pick_model(main_models, "embeddings", args.embedding_model)
    rerank_model = _pick_model(main_models, "rerank", args.rerank_model)

    chat_status, chat_response = _http_json_allow_error(
        f"{args.main_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": main_model,
            "stream": False,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "Reply with exactly PONG."}],
        },
        timeout=600,
    )

    vision_status, vision_response = _http_json_allow_error(
        f"{args.main_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": main_model,
            "stream": False,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the visible text exactly."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        },
        timeout=600,
    )

    embeddings_status, embeddings_response = _http_json_allow_error(
        f"{args.main_url}/v1/embeddings",
        method="POST",
        payload={
            "model": embedding_model,
            "input": "strict agx parity",
        },
    )

    rerank_status, rerank_response = _http_json_allow_error(
        f"{args.main_url}/v1/rerank",
        method="POST",
        payload={
            "model": rerank_model,
            "query": "strict parity",
            "documents": ["strict parity with agx", "irrelevant text"],
        },
        timeout=300,
    )

    first_context_status, first_context_response = _http_json_allow_error(
        f"{args.main_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": main_model,
            "stream": False,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "metadata": {"context_id": args.context_id},
            "messages": [
                {
                    "role": "user",
                    "content": "Remember the code AGX395ONLY and reply with SAVED only.",
                }
            ],
        },
        timeout=600,
    )
    second_context_status, second_context_response = _http_json_allow_error(
        f"{args.main_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": main_model,
            "stream": False,
            "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
            "metadata": {"context_id": args.context_id},
            "messages": [
                {
                    "role": "user",
                    "content": "What code should you remember? Reply with the exact code only.",
                }
            ],
        },
        timeout=600,
    )
    contexts_status, contexts_response = _http_json_allow_error(
        f"{args.main_url}/admin/api/contexts",
    )
    restore_status, restore_response = _http_json_allow_error(
        f"{args.main_url}/admin/api/contexts/{args.context_id}/restore",
        method="POST",
    )
    delete_status, delete_response = _http_json_allow_error(
        f"{args.main_url}/admin/api/contexts/{args.context_id}",
        method="DELETE",
    )

    ocr_status, ocr_response = _http_json_allow_error(
        f"{args.ocr_url}/v1/chat/completions",
        method="POST",
        payload={
            "model": ocr_model,
            "stream": False,
            "temperature": 0,
            "ocr": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read all text in this image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        },
        timeout=600,
    )

    context_items = contexts_response.get("contexts") if isinstance(contexts_response, dict) else []
    context_present = False
    if isinstance(context_items, list):
        for item in context_items:
            if isinstance(item, dict) and item.get("context_id") == args.context_id:
                context_present = True
                break

    summary = {
        "main_models_ok": _contains_all(
            json.dumps(main_models, ensure_ascii=False),
            ["qwen35-35b", "embed-text", "rerank-qwen"],
        ),
        "ocr_models_ok": _contains_all(
            json.dumps(ocr_models, ensure_ascii=False),
            ["ocr-lite"],
        ),
        "chat_ok": chat_status == 200 and "pong" in _assistant_text(chat_response).lower(),
        "vision_ok": vision_status == 200 and _nonempty_assistant_text(vision_response),
        "embeddings_ok": embeddings_status == 200,
        "rerank_ok": rerank_status == 200,
        "contexts_ok": (
            first_context_status == 200
            and second_context_status == 200
            and contexts_status == 200
            and context_present
            and restore_status == 200
            and delete_status == 200
        ),
        "ocr_ok": ocr_status == 200 and _nonempty_assistant_text(ocr_response),
        "duration_sec": round(time.perf_counter() - started, 3),
        "named_context_storage_mode": (
            next(
                (
                    item.get("storage_mode")
                    for item in context_items
                    if isinstance(item, dict) and item.get("context_id") == args.context_id
                ),
                "",
            )
            if isinstance(context_items, list)
            else ""
        ),
        "main_backend_kind": main_runtime.get("capabilities", {})
        .get("backend", {})
        .get("kind"),
    }
    summary["all_ok"] = all(
        bool(summary[key])
        for key in (
            "main_models_ok",
            "ocr_models_ok",
            "chat_ok",
            "vision_ok",
            "embeddings_ok",
            "rerank_ok",
            "contexts_ok",
            "ocr_ok",
        )
    )

    report = {
        "urls": {"main_url": args.main_url, "ocr_url": args.ocr_url},
        "selected_models": {
            "main_model": main_model,
            "ocr_model": ocr_model,
            "embedding_model": embedding_model,
            "rerank_model": rerank_model,
        },
        "main_models": main_models,
        "ocr_models": ocr_models,
        "main_runtime": main_runtime,
        "chat_request": {"status_code": chat_status, "response": chat_response},
        "vision_request": {"status_code": vision_status, "response": vision_response},
        "embeddings_request": {
            "status_code": embeddings_status,
            "response": embeddings_response,
        },
        "rerank_request": {"status_code": rerank_status, "response": rerank_response},
        "named_context": {
            "context_id": args.context_id,
            "first_request": {
                "status_code": first_context_status,
                "response": first_context_response,
            },
            "second_request": {
                "status_code": second_context_status,
                "response": second_context_response,
            },
            "list_request": {
                "status_code": contexts_status,
                "response": contexts_response,
            },
            "restore_request": {
                "status_code": restore_status,
                "response": restore_response,
            },
            "delete_request": {
                "status_code": delete_status,
                "response": delete_response,
            },
        },
        "ocr_request": {"status_code": ocr_status, "response": ocr_response},
        "summary": summary,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not summary["all_ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
