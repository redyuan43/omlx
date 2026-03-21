# SPDX-License-Identifier: Apache-2.0
"""
omlx: LLM inference server, optimized for your Mac.

The MLX-backed engine is optional at import time so pure schema/control-plane
modules remain usable on non-Apple hosts such as Jetson.
"""

from __future__ import annotations

from importlib import import_module

from omlx._version import __version__
from omlx.request import Request, RequestOutput, RequestStatus, SamplingParams

_LAZY_EXPORTS = {
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
    "get_registry": ("omlx.model_registry", "get_registry"),
    "ModelOwnershipError": ("omlx.model_registry", "ModelOwnershipError"),
}

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


def __getattr__(name: str):
    if name == "CacheStats":
        value = __getattr__("PagedCacheStats")
        globals()[name] = value
        return value

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module 'omlx' has no attribute {name!r}")

    module_name, attr_name = target
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("mlx"):
            raise ModuleNotFoundError(
                f"{name} requires the MLX runtime. Install MLX dependencies to use {name}."
            ) from exc
        raise

    value = getattr(module, attr_name)
    globals()[name] = value
    if name == "PagedCacheStats":
        globals()["CacheStats"] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
