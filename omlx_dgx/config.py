# SPDX-License-Identifier: Apache-2.0
"""Configuration and persistence for the DGX runtime/control plane."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def parse_size(size_str: str) -> int:
    """Parse a human-readable size string to bytes."""
    normalized = size_str.strip().upper()
    units = [
        ("TB", 1024**4),
        ("GB", 1024**3),
        ("MB", 1024**2),
        ("KB", 1024),
        ("B", 1),
    ]
    for unit, multiplier in units:
        if normalized.endswith(unit):
            value = float(normalized[: -len(unit)])
            return int(value * multiplier)
    return int(normalized)


@dataclass
class BackendConfig:
    kind: str = "sglang"
    base_url: str = "http://127.0.0.1:30000"
    container_image: str = "nvcr.io/nvidia/tensorrt-llm:latest"
    engine_dir: str = ""
    model_repo_id: str = "Qwen/Qwen3.5-35B-A3B"
    launcher_cmd: str = ""
    runtime_python: str = "python3"
    startup_timeout_seconds: int = 120
    direct_api_enabled: bool = True
    tensor_parallel_size: int = 1
    context_length: int = 65536
    chat_template: Optional[str] = ""
    attention_backend: str = "triton"
    reasoning_parser: str = "qwen3"
    mem_fraction_static: float = 0.80
    trust_remote_code: bool = True
    enable_metrics: bool = True
    enable_cache_report: bool = True
    enable_hierarchical_cache: bool = True
    page_size: int = 64
    hicache_ratio: float = 2.0
    hicache_size: int = 0
    hicache_mem_layout: str = "page_first_direct"
    hicache_io_backend: str = "direct"
    hicache_write_policy: str = "write_through_selective"
    hicache_storage_backend: str = "file"
    hicache_storage_prefetch_policy: str = "timeout"
    hicache_storage_root: str = "~/.omlx-dgx/sglang/hicache"
    hicache_storage_backend_extra_config: Dict[str, Any] = field(
        default_factory=lambda: {
            "prefetch_threshold": 256,
            "prefetch_timeout_base": 0.5,
            "prefetch_timeout_per_ki_token": 0.25,
        }
    )
    admin_api_key: str = "omlx-dgx-admin"


@dataclass
class CacheConfig:
    gpu_max_bytes: int = parse_size("24GB")
    host_max_bytes: int = parse_size("64GB")
    ssd_root: str = "~/.omlx-dgx/cache"
    ssd_max_bytes: int = parse_size("256GB")
    persist_across_restart: bool = True


@dataclass
class ControlPlaneConfig:
    host: str = "127.0.0.1"
    port: int = 8008


@dataclass
class ModelProfile:
    model_id: str
    model_alias: Optional[str] = None
    max_context_window: int = 65536
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    is_default: bool = False

    def api_name(self) -> str:
        return self.model_alias or self.model_id


@dataclass
class DGXRuntimeConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    control_plane: ControlPlaneConfig = field(default_factory=ControlPlaneConfig)
    models: Dict[str, ModelProfile] = field(default_factory=dict)

    def resolve_model_id(self, model_name: Optional[str]) -> Optional[str]:
        if not model_name:
            for model_id, profile in self.models.items():
                if profile.is_default:
                    return model_id
            return next(iter(self.models.keys()), None)

        if model_name in self.models:
            return model_name
        for model_id, profile in self.models.items():
            if profile.model_alias == model_name:
                return model_id
        return model_name

    def public_models(self) -> list[dict[str, Any]]:
        result = []
        for model_id, profile in self.models.items():
            result.append(
                {
                    "id": profile.api_name(),
                    "root": model_id,
                    "max_context_window": profile.max_context_window,
                    "max_tokens": profile.max_tokens,
                    "default": profile.is_default,
                }
            )
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "backend": asdict(self.backend),
            "cache": asdict(self.cache),
            "control_plane": asdict(self.control_plane),
            "models": {
                model_id: asdict(profile) for model_id, profile in self.models.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DGXRuntimeConfig":
        models = {
            model_id: ModelProfile(**profile)
            for model_id, profile in data.get("models", {}).items()
        }
        return cls(
            backend=BackendConfig(**data.get("backend", {})),
            cache=CacheConfig(**data.get("cache", {})),
            control_plane=ControlPlaneConfig(**data.get("control_plane", {})),
            models=models,
        )


class DGXSettingsManager:
    """Atomic JSON-backed settings manager for DGX control-plane state."""

    def __init__(self, base_path: Path) -> None:
        self.base_path = Path(base_path).expanduser().resolve()
        self.settings_path = self.base_path / "settings.json"
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.config = self.load()

    def load(self) -> DGXRuntimeConfig:
        if not self.settings_path.exists():
            return DGXRuntimeConfig()
        data = json.loads(self.settings_path.read_text(encoding="utf-8"))
        return DGXRuntimeConfig.from_dict(data)

    def save(self) -> None:
        payload = json.dumps(self.config.to_dict(), indent=2, sort_keys=True)
        temp_path = self.settings_path.with_suffix(".tmp")
        temp_path.write_text(payload, encoding="utf-8")
        temp_path.replace(self.settings_path)

    def ensure_model(self, profile: ModelProfile) -> None:
        self.config.models[profile.model_id] = profile
        if profile.is_default:
            for model_id, existing in self.config.models.items():
                if model_id != profile.model_id:
                    existing.is_default = False
        self.save()
