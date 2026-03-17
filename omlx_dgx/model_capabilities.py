# SPDX-License-Identifier: Apache-2.0
"""Helpers for inferring model capabilities from model identifiers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class ModelCapabilities:
    embeddings: bool = False
    rerank: bool = False
    vision_chat: bool = False
    ocr: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


def infer_model_capabilities(*refs: Optional[str]) -> ModelCapabilities:
    normalized = " ".join(str(ref or "").lower() for ref in refs if ref)
    embeddings = any(
        token in normalized
        for token in (
            "embedding",
            "embed",
            "nomic-embed",
            "bge-m3",
            "gte-",
            "gte_",
        )
    )
    rerank = any(
        token in normalized
        for token in (
            "rerank",
            "reranker",
            "bge-reranker",
            "qwen3-reranker",
        )
    )
    ocr = any(
        token in normalized
        for token in (
            "ocr",
            "deepseek-ocr",
            "deepseekocr",
            "dots-ocr",
            "dots_ocr",
            "glm-ocr",
            "glm_ocr",
        )
    )
    vision = ocr or any(
        token in normalized
        for token in (
            "-vl",
            "_vl",
            "vision",
            "minicpm-v",
            "minicpm-o",
            "glm-4v",
            "pixtral",
            "qwen3-vl",
            "qwen3_vl",
        )
    )
    return ModelCapabilities(
        embeddings=embeddings,
        rerank=rerank,
        vision_chat=vision,
        ocr=ocr,
    )


def infer_multimodal_capabilities(*refs: Optional[str]) -> ModelCapabilities:
    return infer_model_capabilities(*refs)
