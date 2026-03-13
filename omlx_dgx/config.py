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
    adaptive_backend_base_url: str = ""
    container_image: str = "nvcr.io/nvidia/tensorrt-llm:latest"
    engine_dir: str = ""
    launcher_binary: str = "llama-server"
    model_repo_id: str = "Qwen/Qwen3.5-35B-A3B"
    quant_mode: str = "bf16"
    model_source: str = "hf"
    artifact_path: str = ""
    gguf_variant: str = ""
    launcher_cmd: str = ""
    runtime_python: str = "python3"
    startup_timeout_seconds: int = 120
    direct_api_enabled: bool = True
    tensor_parallel_size: int = 1
    context_length: int = 65536
    ctx_size: int = 16384
    parallel_slots: int = 1
    n_gpu_layers: int = 999
    flash_attn: bool = True
    batch_size: int = 2048
    ubatch_size: int = 512
    cache_ram_mib: int = 8192
    cache_reuse: int = 0
    checkpoint_every_n_tokens: int = 8192
    ctx_checkpoints: int = 32
    slot_prompt_similarity: float = 0.10
    enable_runtime_metrics: bool = False
    enable_session_stickiness: bool = True
    sticky_session_prompt_threshold: int = 2048
    sticky_max_sessions: int = 256
    split_mode: str = "row"
    no_context_shift: bool = True
    jinja: bool = True
    reasoning_format: str = "deepseek"
    chat_template: Optional[str] = ""
    attention_backend: str = "triton"
    reasoning_parser: str = "qwen3"
    mamba_ssm_dtype: str = ""
    mem_fraction_static: float = 0.80
    disable_cuda_graph: bool = False
    prefill_strategy: str = "fixed"
    chunked_prefill_size: int = 8192
    fixed_chunked_prefill_size: int = 8192
    adaptive_short_prompt_threshold: int = 2048
    adaptive_long_context_chunk_size: int = 8192
    adaptive_repeat_prefix_chunk_size: int = 1024
    adaptive_max_sticky_sessions: int = 256
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
            "hicache_storage_pass_prefix_keys": True,
        }
    )
    admin_api_key: str = "omlx-dgx-admin"

    def __post_init__(self) -> None:
        if self.quant_mode not in {
            "bf16",
            "awq_int4",
            "awq_marlin_int4",
            "gguf_experimental",
            "lmstudio_baseline",
        }:
            self.quant_mode = "bf16"
        if self.model_source not in {"hf", "gguf", "lmstudio_api"}:
            self.model_source = "hf"
        if self.split_mode not in {"none", "layer", "row"}:
            self.split_mode = "row"
        if self.mamba_ssm_dtype not in {"", "float32", "bfloat16", "float16"}:
            self.mamba_ssm_dtype = ""
        if self.quant_mode == "gguf_experimental":
            self.model_source = "gguf"
        elif self.quant_mode == "lmstudio_baseline" and self.model_source == "hf":
            self.model_source = "lmstudio_api"
        if not self.fixed_chunked_prefill_size:
            self.fixed_chunked_prefill_size = self.chunked_prefill_size or 8192
        if not self.chunked_prefill_size:
            self.chunked_prefill_size = self.fixed_chunked_prefill_size
        if not self.adaptive_long_context_chunk_size:
            self.adaptive_long_context_chunk_size = self.fixed_chunked_prefill_size
        if self.prefill_strategy not in {"fixed", "adaptive"}:
            self.prefill_strategy = "fixed"


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
        backend_data = dict(data.get("backend", {}))
        legacy_chunk_size = backend_data.get("chunked_prefill_size")
        if "fixed_chunked_prefill_size" not in backend_data and legacy_chunk_size:
            backend_data["fixed_chunked_prefill_size"] = legacy_chunk_size
        if "adaptive_long_context_chunk_size" not in backend_data:
            backend_data["adaptive_long_context_chunk_size"] = backend_data.get(
                "fixed_chunked_prefill_size",
                legacy_chunk_size or 8192,
            )
        if "chunked_prefill_size" not in backend_data and "fixed_chunked_prefill_size" in backend_data:
            backend_data["chunked_prefill_size"] = backend_data["fixed_chunked_prefill_size"]
        models = {
            model_id: ModelProfile(**profile)
            for model_id, profile in data.get("models", {}).items()
        }
        return cls(
            backend=BackendConfig(**backend_data),
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
