#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Register AGX parity auxiliary models against the AMD DGX control plane."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


def _http_json(session: requests.Session, method: str, url: str, **kwargs):
    response = session.request(method, url, timeout=30, **kwargs)
    response.raise_for_status()
    if response.content:
        return response.json()
    return {}


def _read_manifest_value(manifest_path: str, *keys: str) -> str:
    if not manifest_path:
        return ""
    payload = json.loads(Path(manifest_path).expanduser().resolve().read_text(encoding="utf-8"))
    current = payload
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key)
    return "" if current is None else str(current)


def main() -> None:
    parser = argparse.ArgumentParser(description="Register AMD AGX parity auxiliary models")
    parser.add_argument("--control-plane-url", default="http://127.0.0.1:8008")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--embedding-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--embedding-model-name", default="nomic-embed-text")
    parser.add_argument("--embedding-model-id", default="embed-text")
    parser.add_argument("--embedding-model-alias", default="embed-text")
    parser.add_argument("--rerank-artifact-path", default="")
    parser.add_argument("--rerank-base-url", default="http://127.0.0.1:30030")
    parser.add_argument("--rerank-model-id", default="rerank-qwen")
    parser.add_argument("--rerank-model-alias", default="rerank-qwen")
    parser.add_argument("--rerank-ctx-size", type=int, default=8192)
    parser.add_argument("--rerank-parallel-slots", type=int, default=1)
    args = parser.parse_args()

    rerank_artifact_path = (
        args.rerank_artifact_path
        or _read_manifest_value(args.manifest, "rerank", "artifact_path")
    )
    if not rerank_artifact_path:
        raise SystemExit("rerank artifact path is required")
    rerank_artifact = Path(rerank_artifact_path).expanduser().resolve()
    if not rerank_artifact.exists():
        raise SystemExit(f"rerank artifact not found: {rerank_artifact}")

    session = requests.Session()
    _http_json(session, "GET", f"{args.control_plane_url.rstrip('/')}/health")

    registrations = [
        {
            "model_id": args.embedding_model_id,
            "model_alias": args.embedding_model_alias,
            "backend_kind": "openai_compatible",
            "base_url": args.embedding_base_url,
            "backend_model_name": args.embedding_model_name,
            "supports_embeddings": True,
            "primary_service": "embeddings",
        },
        {
            "model_id": args.rerank_model_id,
            "model_alias": args.rerank_model_alias,
            "backend_kind": "llama_cpp",
            "artifact_path": str(rerank_artifact),
            "base_url": args.rerank_base_url,
            "ctx_size": args.rerank_ctx_size,
            "parallel_slots": args.rerank_parallel_slots,
            "supports_rerank": True,
            "primary_service": "rerank",
        },
    ]

    results = []
    for payload in registrations:
        response = _http_json(
            session,
            "POST",
            f"{args.control_plane_url.rstrip('/')}/admin/api/runtime/model-pool",
            json=payload,
        )
        results.append({"request": payload, "response": response})

    public_models = _http_json(
        session,
        "GET",
        f"{args.control_plane_url.rstrip('/')}/v1/models",
    )
    print(
        json.dumps(
            {
                "control_plane_url": args.control_plane_url,
                "registrations": results,
                "public_models": public_models,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
