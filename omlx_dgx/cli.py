# SPDX-License-Identifier: Apache-2.0
"""CLI for the experimental DGX runtime/control plane."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from omlx_dgx.config import (
    LLAMA_CPP_SERVING_PRESETS,
    DGXSettingsManager,
    ModelProfile,
    apply_llama_cpp_serving_preset,
)
from omlx_dgx.control_plane.app import create_app


def serve_command(args: argparse.Namespace) -> None:
    settings = DGXSettingsManager(Path(args.base_path))
    if args.backend_kind:
        settings.config.backend.kind = args.backend_kind
    if args.backend_url:
        settings.config.backend.base_url = args.backend_url
    if args.host:
        settings.config.control_plane.host = args.host
    if args.port:
        settings.config.control_plane.port = args.port
    if args.runtime_python:
        settings.config.backend.runtime_python = args.runtime_python
    if args.launcher_binary:
        settings.config.backend.launcher_binary = args.launcher_binary
    if args.launcher_cmd is not None:
        settings.config.backend.launcher_cmd = args.launcher_cmd
    if args.startup_timeout_seconds is not None:
        settings.config.backend.startup_timeout_seconds = args.startup_timeout_seconds
    if args.model_repo_id:
        settings.config.backend.model_repo_id = args.model_repo_id
    if args.quant_mode:
        settings.config.backend.quant_mode = args.quant_mode
    if args.model_source:
        settings.config.backend.model_source = args.model_source
    if args.artifact_path is not None:
        settings.config.backend.artifact_path = args.artifact_path
    if args.gguf_variant is not None:
        settings.config.backend.gguf_variant = args.gguf_variant
    if args.serving_preset is not None:
        apply_llama_cpp_serving_preset(settings.config.backend, args.serving_preset)
    if args.ctx_size is not None:
        settings.config.backend.ctx_size = args.ctx_size
    if args.parallel_slots is not None:
        settings.config.backend.parallel_slots = args.parallel_slots
    if args.n_gpu_layers is not None:
        settings.config.backend.n_gpu_layers = args.n_gpu_layers
    if args.flash_attn is not None:
        settings.config.backend.flash_attn = args.flash_attn
    if args.batch_size is not None:
        settings.config.backend.batch_size = args.batch_size
    if args.ubatch_size is not None:
        settings.config.backend.ubatch_size = args.ubatch_size
    if args.cache_ram_mib is not None:
        settings.config.backend.cache_ram_mib = args.cache_ram_mib
    if args.cache_reuse is not None:
        settings.config.backend.cache_reuse = args.cache_reuse
    if args.checkpoint_every_n_tokens is not None:
        settings.config.backend.checkpoint_every_n_tokens = args.checkpoint_every_n_tokens
    if args.ctx_checkpoints is not None:
        settings.config.backend.ctx_checkpoints = args.ctx_checkpoints
    if args.slot_prompt_similarity is not None:
        settings.config.backend.slot_prompt_similarity = args.slot_prompt_similarity
    if args.enable_runtime_metrics is not None:
        settings.config.backend.enable_runtime_metrics = args.enable_runtime_metrics
    if args.enable_session_stickiness is not None:
        settings.config.backend.enable_session_stickiness = args.enable_session_stickiness
    if args.sticky_session_prompt_threshold is not None:
        settings.config.backend.sticky_session_prompt_threshold = (
            args.sticky_session_prompt_threshold
        )
    if args.sticky_max_sessions is not None:
        settings.config.backend.sticky_max_sessions = args.sticky_max_sessions
    if args.split_mode:
        settings.config.backend.split_mode = args.split_mode
    if args.no_context_shift is not None:
        settings.config.backend.no_context_shift = args.no_context_shift
    if args.jinja is not None:
        settings.config.backend.jinja = args.jinja
    if args.reasoning_format is not None:
        settings.config.backend.reasoning_format = args.reasoning_format
    if args.mamba_ssm_dtype:
        settings.config.backend.mamba_ssm_dtype = args.mamba_ssm_dtype
    if args.disable_cuda_graph is not None:
        settings.config.backend.disable_cuda_graph = args.disable_cuda_graph
    if args.prefill_strategy:
        settings.config.backend.prefill_strategy = args.prefill_strategy
    if args.fixed_chunked_prefill_size is not None:
        settings.config.backend.fixed_chunked_prefill_size = args.fixed_chunked_prefill_size
        settings.config.backend.chunked_prefill_size = args.fixed_chunked_prefill_size
    if args.chunked_prefill_size is not None:
        settings.config.backend.fixed_chunked_prefill_size = args.chunked_prefill_size
        settings.config.backend.chunked_prefill_size = args.chunked_prefill_size
    if args.adaptive_short_prompt_threshold is not None:
        settings.config.backend.adaptive_short_prompt_threshold = (
            args.adaptive_short_prompt_threshold
        )
    if args.adaptive_long_context_chunk_size is not None:
        settings.config.backend.adaptive_long_context_chunk_size = (
            args.adaptive_long_context_chunk_size
        )
    if args.adaptive_repeat_prefix_chunk_size is not None:
        settings.config.backend.adaptive_repeat_prefix_chunk_size = (
            args.adaptive_repeat_prefix_chunk_size
        )
    if args.adaptive_backend_url:
        settings.config.backend.adaptive_backend_base_url = args.adaptive_backend_url
    if args.enable_hierarchical_cache is not None:
        settings.config.backend.enable_hierarchical_cache = (
            args.enable_hierarchical_cache
        )
    if args.hicache_size_gb is not None:
        settings.config.backend.hicache_size = args.hicache_size_gb
    if args.hicache_pass_prefix_keys is not None:
        settings.config.backend.hicache_storage_backend_extra_config[
            "hicache_storage_pass_prefix_keys"
        ] = args.hicache_pass_prefix_keys
    settings.config.backend.__post_init__()
    if args.model_id:
        settings.ensure_model(
            ModelProfile(
                model_id=args.model_id,
                model_alias=args.model_alias,
                is_default=True,
            )
        )
    else:
        settings.save()

    app = create_app(base_path=args.base_path, settings_manager=settings)
    uvicorn.run(
        app,
        host=settings.config.control_plane.host,
        port=settings.config.control_plane.port,
        log_level="info",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experimental DGX control plane")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the DGX control plane")
    serve.add_argument("--base-path", default="~/.omlx-dgx")
    serve.add_argument("--backend-kind", default=None)
    serve.add_argument("--backend-url", default=None)
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--runtime-python", default=None)
    serve.add_argument("--launcher-binary", default=None)
    serve.add_argument("--launcher-cmd", default=None)
    serve.add_argument("--startup-timeout-seconds", type=int, default=None)
    serve.add_argument("--model-id", default="")
    serve.add_argument("--model-alias", default=None)
    serve.add_argument("--model-repo-id", default="")
    serve.add_argument(
        "--quant-mode",
        choices=(
            "bf16",
            "awq_int4",
            "awq_marlin_int4",
            "gguf_experimental",
            "lmstudio_baseline",
        ),
        default=None,
    )
    serve.add_argument(
        "--model-source",
        choices=("hf", "gguf", "lmstudio_api"),
        default=None,
    )
    serve.add_argument("--artifact-path", default=None)
    serve.add_argument("--gguf-variant", default=None)
    serve.add_argument(
        "--serving-preset",
        choices=tuple(sorted(LLAMA_CPP_SERVING_PRESETS)),
        default=None,
    )
    serve.add_argument("--ctx-size", type=int, default=None)
    serve.add_argument("--parallel-slots", type=int, default=None)
    serve.add_argument("--n-gpu-layers", type=int, default=None)
    serve.add_argument(
        "--flash-attn",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--batch-size", type=int, default=None)
    serve.add_argument("--ubatch-size", type=int, default=None)
    serve.add_argument("--cache-ram-mib", type=int, default=None)
    serve.add_argument("--cache-reuse", type=int, default=None)
    serve.add_argument("--checkpoint-every-n-tokens", type=int, default=None)
    serve.add_argument("--ctx-checkpoints", type=int, default=None)
    serve.add_argument("--slot-prompt-similarity", type=float, default=None)
    serve.add_argument(
        "--enable-runtime-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument(
        "--enable-session-stickiness",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--sticky-session-prompt-threshold", type=int, default=None)
    serve.add_argument("--sticky-max-sessions", type=int, default=None)
    serve.add_argument("--split-mode", choices=("none", "layer", "row"), default=None)
    serve.add_argument("--no-context-shift", dest="no_context_shift", action="store_true")
    serve.add_argument("--allow-context-shift", dest="no_context_shift", action="store_false")
    serve.add_argument(
        "--jinja",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--reasoning-format", default=None)
    serve.add_argument(
        "--mamba-ssm-dtype",
        choices=("float32", "bfloat16", "float16"),
        default=None,
    )
    serve.add_argument(
        "--disable-cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--prefill-strategy", choices=("fixed", "adaptive"), default=None)
    serve.add_argument("--chunked-prefill-size", type=int, default=None)
    serve.add_argument("--fixed-chunked-prefill-size", type=int, default=None)
    serve.add_argument("--adaptive-short-prompt-threshold", type=int, default=None)
    serve.add_argument("--adaptive-long-context-chunk-size", type=int, default=None)
    serve.add_argument("--adaptive-repeat-prefix-chunk-size", type=int, default=None)
    serve.add_argument("--adaptive-backend-url", default=None)
    serve.add_argument(
        "--enable-hierarchical-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.add_argument("--hicache-size-gb", type=int, default=None)
    serve.add_argument(
        "--hicache-pass-prefix-keys",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    serve.set_defaults(func=serve_command)
    serve.set_defaults(no_context_shift=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
