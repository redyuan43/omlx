# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac.

This package provides native Apple Silicon GPU acceleration using
Apple's MLX framework and mlx-lm for LLMs.

The public package exports stay stable, but heavyweight MLX-backed
components are imported lazily so Linux-only utilities such as
``omlx_dgx`` can reuse ``omlx.api`` models without importing ``mlx``.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from omlx._version import __version__

if TYPE_CHECKING:
    from omlx.cache.paged_cache import BlockTable, CacheBlock, PagedCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.cache.stats import PagedCacheStats, PrefixCacheStats
    from omlx.engine_core import AsyncEngineCore, EngineConfig, EngineCore
    from omlx.model_registry import ModelOwnershipError, get_registry
    from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams
    from omlx.scheduler import Scheduler, SchedulerConfig, SchedulerOutput


_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "Request": ("omlx.request", "Request"),
    "RequestOutput": ("omlx.request", "RequestOutput"),
    "RequestStatus": ("omlx.request", "RequestStatus"),
    "SamplingParams": ("omlx.request", "SamplingParams"),
    "Scheduler": ("omlx.scheduler", "Scheduler"),
    "SchedulerConfig": ("omlx.scheduler", "SchedulerConfig"),
    "SchedulerOutput": ("omlx.scheduler", "SchedulerOutput"),
    "EngineCore": ("omlx.engine_core", "EngineCore"),
    "AsyncEngineCore": ("omlx.engine_core", "AsyncEngineCore"),
    "EngineConfig": ("omlx.engine_core", "EngineConfig"),
    "BlockAwarePrefixCache": ("omlx.cache.prefix_cache", "BlockAwarePrefixCache"),
    "PagedCacheManager": ("omlx.cache.paged_cache", "PagedCacheManager"),
    "CacheBlock": ("omlx.cache.paged_cache", "CacheBlock"),
    "BlockTable": ("omlx.cache.paged_cache", "BlockTable"),
    "PrefixCacheStats": ("omlx.cache.stats", "PrefixCacheStats"),
    "PagedCacheStats": ("omlx.cache.stats", "PagedCacheStats"),
    "CacheStats": ("omlx.cache.stats", "PagedCacheStats"),
    "get_registry": ("omlx.model_registry", "get_registry"),
    "ModelOwnershipError": ("omlx.model_registry", "ModelOwnershipError"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    if name in {"PagedCacheStats", "CacheStats"}:
        globals()["CacheStats"] = value
        globals()["PagedCacheStats"] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    "Request",
    "RequestOutput",
    "RequestStatus",
    "SamplingParams",
    "Scheduler",
    "SchedulerConfig",
    "SchedulerOutput",
    "EngineCore",
    "AsyncEngineCore",
    "EngineConfig",
    "get_registry",
    "ModelOwnershipError",
    "BlockAwarePrefixCache",
    "PagedCacheManager",
    "CacheBlock",
    "BlockTable",
    "PrefixCacheStats",
    "PagedCacheStats",
    "CacheStats",
    "__version__",
]
