#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Download the AMD baseline GGUF artifacts from Hugging Face."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


@dataclass(frozen=True)
class ModelRequest:
    name: str
    repo_id: str
    variant: str


MODEL_REQUESTS = (
    ModelRequest(
        name="chat",
        repo_id="lmstudio-community/Qwen3.5-4B-GGUF",
        variant="Q4_K_M",
    ),
    ModelRequest(
        name="rerank",
        repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        variant="Q8_0",
    ),
)


def _repo_dir(root: Path, repo_id: str) -> Path:
    return root / repo_id.rsplit("/", 1)[-1]


def _pick_gguf_file(api: HfApi, request: ModelRequest) -> str:
    files = api.list_repo_files(request.repo_id, repo_type="model")
    candidates = []
    for filename in files:
        basename = Path(filename).name
        if not basename.endswith(".gguf"):
            continue
        if "mmproj" in basename.lower():
            continue
        if request.variant.upper() not in basename.upper():
            continue
        candidates.append(filename)
    if len(candidates) != 1:
        raise RuntimeError(
            f"{request.repo_id} expected exactly one GGUF for {request.variant}, found {candidates!r}"
        )
    return candidates[0]


def _download_request(api: HfApi, request: ModelRequest, output_root: Path) -> dict[str, str]:
    filename = _pick_gguf_file(api, request)
    local_dir = _repo_dir(output_root, request.repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=request.repo_id,
        repo_type="model",
        allow_patterns=[filename],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    local_path = local_dir / filename
    if not local_path.exists():
        raise RuntimeError(f"download did not produce {local_path}")
    return {
        "repo_id": request.repo_id,
        "variant": request.variant,
        "filename": filename,
        "path": str(local_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download AMD baseline GGUF models")
    parser.add_argument(
        "--output-dir",
        default="~/.cache/omlx-dgx-models",
        help="Root directory for downloaded GGUF artifacts",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    downloaded = {
        request.name: _download_request(api, request, output_dir)
        for request in MODEL_REQUESTS
    }
    print(json.dumps({"output_dir": str(output_dir), "models": downloaded}, indent=2))


if __name__ == "__main__":
    main()
