#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Resolve or download the AMD AGX parity model assets."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class RepoAsset:
    repo_id: str
    model_variant: str
    mmproj_variant: str = ""


SEARCH_ROOTS = (
    Path("~/.lmstudio/models").expanduser(),
    Path("~/.cache/omlx-dgx-models").expanduser(),
)

MAIN_MATCHES = (
    "Qwen3.5-35B-A3B-Q4_K_M.gguf",
    "Qwen3.5-35B-A3B.Q4_K_M.gguf",
)
MAIN_MMPROJ_MATCHES = (
    "mmproj-Qwen3.5-35B-A3B-BF16.gguf",
    "Qwen3.5-35B-A3B.mmproj-Q8_0.gguf",
    "Qwen3.5-35B-A3B.mmproj.gguf",
)
RERANK_REPO = RepoAsset(
    repo_id="ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
    model_variant="Q8_0",
)
OCR_REPO = RepoAsset(
    repo_id="mradermacher/GLM-OCR-GGUF",
    model_variant="Q4_K_M",
    mmproj_variant="mmproj-Q8_0",
)


def _iter_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if not root.exists():
            continue
        yield from root.rglob("*.gguf")


def _first_matching_file(
    roots: Sequence[Path],
    *,
    exact_names: Sequence[str] = (),
    include_tokens: Sequence[str] = (),
    exclude_tokens: Sequence[str] = (),
) -> Path | None:
    normalized_exact = {name.lower() for name in exact_names}
    normalized_include = tuple(token.lower() for token in include_tokens if token)
    normalized_exclude = tuple(token.lower() for token in exclude_tokens if token)
    for path in _iter_files(roots):
        name = path.name.lower()
        if normalized_exact and name in normalized_exact:
            return path
        if normalized_include and all(token in name for token in normalized_include):
            if any(token in name for token in normalized_exclude):
                continue
            return path
    return None


def _repo_dir(output_root: Path, repo_id: str) -> Path:
    return output_root / repo_id.rsplit("/", 1)[-1]


def _pick_repo_file(api, repo_id: str, *, variant: str, want_mmproj: bool) -> str:
    candidates = []
    for filename in api.list_repo_files(repo_id, repo_type="model"):
        basename = Path(filename).name
        lowered = basename.lower()
        is_mmproj = "mmproj" in lowered
        if is_mmproj != want_mmproj:
            continue
        if want_mmproj:
            if variant.lower() not in lowered:
                continue
        elif variant.upper() not in basename.upper():
            continue
        if not basename.endswith(".gguf"):
            continue
        candidates.append(filename)
    if len(candidates) != 1:
        raise RuntimeError(
            f"{repo_id} expected exactly one {'mmproj' if want_mmproj else 'model'} file for {variant}, found {candidates!r}"
        )
    return candidates[0]


def _download_repo_file(api, repo_id: str, filename: str, output_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    local_dir = _repo_dir(output_root, repo_id)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=[filename],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    local_path = local_dir / filename
    if not local_path.exists():
        raise RuntimeError(f"download did not produce {local_path}")
    return local_path


def _resolve_repo_asset(
    asset: RepoAsset,
    *,
    output_root: Path,
    download_missing: bool,
    want_mmproj: bool,
) -> Path | None:
    from huggingface_hub import HfApi

    api = HfApi()
    variant = asset.mmproj_variant if want_mmproj else asset.model_variant
    filename = _pick_repo_file(
        api,
        asset.repo_id,
        variant=variant,
        want_mmproj=want_mmproj,
    )
    if not download_missing:
        return None
    return _download_repo_file(api, asset.repo_id, filename, output_root)


def _ensure_ollama_model(model_name: str, *, pull: bool) -> dict:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return {
            "model_name": model_name,
            "available": False,
            "checked": False,
            "error": str(exc),
        }
    available = any(
        line.split()[0] == model_name
        for line in result.stdout.splitlines()
        if line.strip() and not line.startswith("NAME")
    )
    pulled = False
    if not available and pull:
        subprocess.run(["ollama", "pull", model_name], check=True)
        available = True
        pulled = True
    return {
        "model_name": model_name,
        "available": available,
        "checked": True,
        "pulled": pulled,
    }


def _resolve_required_path(
    override: str,
    *,
    local_matcher,
    remote_asset: RepoAsset | None,
    output_root: Path,
    download_missing: bool,
    want_mmproj: bool = False,
) -> Path | None:
    if override:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"requested artifact does not exist: {path}")
        return path
    local_path = local_matcher()
    if local_path is not None:
        return local_path.resolve()
    if remote_asset is None:
        return None
    return _resolve_repo_asset(
        remote_asset,
        output_root=output_root,
        download_missing=download_missing,
        want_mmproj=want_mmproj,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve AMD AGX parity model assets")
    parser.add_argument("--output", default="")
    parser.add_argument("--output-dir", default="~/.cache/omlx-dgx-models")
    parser.add_argument("--download-missing", action="store_true")
    parser.add_argument("--pull-embedding", action="store_true")
    parser.add_argument("--main-artifact-path", default="")
    parser.add_argument("--main-mmproj-path", default="")
    parser.add_argument("--rerank-artifact-path", default="")
    parser.add_argument("--ocr-artifact-path", default="")
    parser.add_argument("--ocr-mmproj-path", default="")
    parser.add_argument("--embedding-model-name", default="nomic-embed-text")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    main_artifact = _resolve_required_path(
        args.main_artifact_path,
        local_matcher=lambda: _first_matching_file(
            SEARCH_ROOTS,
            exact_names=MAIN_MATCHES,
        ),
        remote_asset=None,
        output_root=output_dir,
        download_missing=args.download_missing,
    )
    main_mmproj = _resolve_required_path(
        args.main_mmproj_path,
        local_matcher=lambda: _first_matching_file(
            SEARCH_ROOTS,
            exact_names=MAIN_MMPROJ_MATCHES,
            include_tokens=("qwen3.5-35b-a3b", "mmproj"),
        ),
        remote_asset=None,
        output_root=output_dir,
        download_missing=args.download_missing,
    )
    rerank_artifact = _resolve_required_path(
        args.rerank_artifact_path,
        local_matcher=lambda: _first_matching_file(
            SEARCH_ROOTS,
            include_tokens=("qwen3", "reranker", "q8_0"),
            exclude_tokens=("mmproj",),
        ),
        remote_asset=RERANK_REPO,
        output_root=output_dir,
        download_missing=args.download_missing,
    )
    ocr_artifact = _resolve_required_path(
        args.ocr_artifact_path,
        local_matcher=lambda: _first_matching_file(
            SEARCH_ROOTS,
            include_tokens=("glm-ocr", "q4_k_m"),
            exclude_tokens=("mmproj",),
        ),
        remote_asset=OCR_REPO,
        output_root=output_dir,
        download_missing=args.download_missing,
    )
    ocr_mmproj = _resolve_required_path(
        args.ocr_mmproj_path,
        local_matcher=lambda: _first_matching_file(
            SEARCH_ROOTS,
            include_tokens=("glm-ocr", "mmproj"),
        ),
        remote_asset=OCR_REPO,
        output_root=output_dir,
        download_missing=args.download_missing,
        want_mmproj=True,
    )
    embedding = _ensure_ollama_model(
        args.embedding_model_name,
        pull=args.pull_embedding,
    )

    manifest = {
        "output_dir": str(output_dir),
        "chat": {
            "model_id": "qwen35-35b",
            "artifact_path": "" if main_artifact is None else str(main_artifact),
            "mmproj_path": "" if main_mmproj is None else str(main_mmproj),
        },
        "rerank": {
            "model_id": "rerank-qwen",
            "artifact_path": "" if rerank_artifact is None else str(rerank_artifact),
        },
        "ocr": {
            "model_id": "ocr-lite",
            "artifact_path": "" if ocr_artifact is None else str(ocr_artifact),
            "mmproj_path": "" if ocr_mmproj is None else str(ocr_mmproj),
        },
        "embedding": embedding,
    }
    encoded = json.dumps(manifest, indent=2)
    if args.output:
        Path(args.output).expanduser().resolve().write_text(encoded, encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
