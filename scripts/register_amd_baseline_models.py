#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Register the AMD baseline auxiliary models against the DGX control plane."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests


@dataclass(frozen=True)
class LMStudioMatch:
    model_id: str
    model_alias: str
    required_tokens: tuple[str, ...]
    supports_embeddings: bool = False
    supports_vision: bool = False
    supports_ocr: bool = False


LMSTUDIO_MATCHES = (
    LMStudioMatch(
        model_id="nomic-embed",
        model_alias="nomic-embed",
        required_tokens=("nomic", "embed"),
        supports_embeddings=True,
    ),
    LMStudioMatch(
        model_id="qwen3-vl-4b",
        model_alias="qwen3-vl-4b",
        required_tokens=("qwen3-vl-4b",),
        supports_vision=True,
    ),
    LMStudioMatch(
        model_id="glm-ocr",
        model_alias="glm-ocr",
        required_tokens=("glm-ocr",),
        supports_vision=True,
        supports_ocr=True,
    ),
)


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _http_json(session: requests.Session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    if response.content:
        return response.json()
    return {}


def _fetch_model_ids(session: requests.Session, base_url: str) -> list[str]:
    payload = _http_json(session, "GET", f"{base_url.rstrip('/')}/v1/models")
    ids = []
    for item in payload.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def _find_model_id(
    model_ids: Iterable[str],
    *,
    explicit: str,
    required_tokens: tuple[str, ...],
) -> str:
    candidates = list(model_ids)
    if explicit:
        if explicit not in candidates:
            raise RuntimeError(f"LM Studio model id not found: {explicit}")
        return explicit
    normalized_candidates = [(_normalize(item), item) for item in candidates]
    matches = [
        original
        for normalized, original in normalized_candidates
        if all(token in normalized for token in required_tokens)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one LM Studio model for tokens {required_tokens!r}, found {matches!r}"
        )
    return matches[0]


def _register_model(session: requests.Session, control_plane_url: str, payload: dict) -> dict:
    return _http_json(
        session,
        "POST",
        f"{control_plane_url.rstrip('/')}/admin/api/runtime/model-pool",
        json=payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Register AMD baseline capability models")
    parser.add_argument("--control-plane-url", default="http://127.0.0.1:8010")
    parser.add_argument("--lmstudio-url", default="http://127.0.0.1:1234")
    parser.add_argument("--rerank-artifact-path", required=True)
    parser.add_argument("--rerank-base-url", default="http://127.0.0.1:31020")
    parser.add_argument("--rerank-model-id", default="qwen35-reranker")
    parser.add_argument("--rerank-model-alias", default="qwen35-reranker")
    parser.add_argument("--rerank-ctx-size", type=int, default=8192)
    parser.add_argument("--rerank-parallel-slots", type=int, default=1)
    parser.add_argument("--embedding-model-name", default="")
    parser.add_argument("--vision-model-name", default="")
    parser.add_argument("--ocr-model-name", default="")
    parser.add_argument("--skip-lmstudio", action="store_true")
    args = parser.parse_args()

    rerank_artifact_path = Path(args.rerank_artifact_path).expanduser().resolve()
    if not rerank_artifact_path.exists():
        raise SystemExit(f"Rerank artifact not found: {rerank_artifact_path}")

    session = requests.Session()
    _http_json(session, "GET", f"{args.control_plane_url.rstrip('/')}/health")
    resolved_lmstudio_models: dict[str, str] = {}
    if not args.skip_lmstudio:
        lmstudio_models = _fetch_model_ids(session, args.lmstudio_url)
        explicit_names = {
            "nomic-embed": args.embedding_model_name,
            "qwen3-vl-4b": args.vision_model_name,
            "glm-ocr": args.ocr_model_name,
        }
        for match in LMSTUDIO_MATCHES:
            resolved_lmstudio_models[match.model_id] = _find_model_id(
                lmstudio_models,
                explicit=explicit_names[match.model_id],
                required_tokens=match.required_tokens,
            )

    registrations = []
    registrations.append(
        {
            "model_id": args.rerank_model_id,
            "model_alias": args.rerank_model_alias,
            "backend_kind": "llama_cpp",
            "artifact_path": str(rerank_artifact_path),
            "base_url": args.rerank_base_url,
            "ctx_size": args.rerank_ctx_size,
            "parallel_slots": args.rerank_parallel_slots,
            "supports_rerank": True,
        }
    )
    if not args.skip_lmstudio:
        for match in LMSTUDIO_MATCHES:
            registrations.append(
                {
                    "model_id": match.model_id,
                    "model_alias": match.model_alias,
                    "backend_kind": "openai_compatible",
                    "base_url": args.lmstudio_url,
                    "backend_model_name": resolved_lmstudio_models[match.model_id],
                    "supports_embeddings": match.supports_embeddings,
                    "supports_vision": match.supports_vision,
                    "supports_ocr": match.supports_ocr,
                }
            )

    results = []
    for payload in registrations:
        results.append(
            {
                "request": payload,
                "response": _register_model(session, args.control_plane_url, payload),
            }
        )

    public_models = _http_json(
        session,
        "GET",
        f"{args.control_plane_url.rstrip('/')}/v1/models",
    )
    print(
        json.dumps(
            {
                "control_plane_url": args.control_plane_url,
                "lmstudio_url": args.lmstudio_url,
                "resolved_lmstudio_models": resolved_lmstudio_models,
                "registrations": results,
                "public_models": public_models,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
