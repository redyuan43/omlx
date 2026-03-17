# SPDX-License-Identifier: Apache-2.0
"""Helpers for inferring multimodal model capabilities from model identifiers."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass(frozen=True)
class MultimodalCapabilities:
    vision_chat: bool = False
    ocr: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return asdict(self)


def infer_multimodal_capabilities(*refs: Optional[str]) -> MultimodalCapabilities:
    normalized = " ".join(str(ref or "").lower() for ref in refs if ref)
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
        )
    )
    return MultimodalCapabilities(vision_chat=vision, ocr=ocr)
