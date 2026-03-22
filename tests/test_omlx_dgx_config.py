# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import omlx_dgx.cli as cli
from omlx_dgx.config import (
    DGXSettingsManager,
    LlamaCppModelRegistration,
    ModelProfile,
    RECOMMENDED_LLAMA_CPP_GGUF_MODEL_REPO_ID,
    RECOMMENDED_LLAMA_CPP_GGUF_VARIANT,
)
from omlx_dgx.tiered_kv import PersistentManifestStore, StoredBlockRecord


def test_settings_manager_persists_default_model(tmp_path: Path):
    manager = DGXSettingsManager(tmp_path)
    manager.ensure_model(ModelProfile(model_id="qwen35-35b", model_alias="qwen35", is_default=True))

    reloaded = DGXSettingsManager(tmp_path)
    assert reloaded.config.resolve_model_id("qwen35") == "qwen35-35b"
    assert reloaded.config.resolve_model_id(None) == "qwen35-35b"
    assert reloaded.config.backend.kind == "sglang"
    assert reloaded.config.backend.model_repo_id == "Qwen/Qwen3.5-35B-A3B"
    assert reloaded.config.backend.quant_mode == "bf16"
    assert reloaded.config.backend.model_source == "hf"
    assert reloaded.config.backend.artifact_path == ""
    assert reloaded.config.backend.gguf_variant == ""
    assert reloaded.config.backend.launcher_binary == "llama-server"
    assert reloaded.config.backend.attention_backend == "triton"
    assert reloaded.config.backend.mamba_ssm_dtype == ""
    assert reloaded.config.backend.chat_template == ""
    assert reloaded.config.backend.ctx_size == 16384
    assert reloaded.config.backend.parallel_slots == 1
    assert reloaded.config.backend.n_gpu_layers == 999
    assert reloaded.config.backend.flash_attn is True
    assert reloaded.config.backend.batch_size == 2048
    assert reloaded.config.backend.ubatch_size == 512
    assert reloaded.config.backend.cache_ram_mib == 8192
    assert reloaded.config.backend.cache_reuse == 0
    assert reloaded.config.backend.checkpoint_every_n_tokens == 8192
    assert reloaded.config.backend.ctx_checkpoints == 32
    assert reloaded.config.backend.slot_prompt_similarity == 0.10
    assert reloaded.config.backend.enable_runtime_metrics is False
    assert reloaded.config.backend.enable_session_stickiness is True
    assert reloaded.config.backend.enable_session_restore is True
    assert reloaded.config.backend.session_restore_min_prompt_tokens == 0
    assert reloaded.config.backend.sticky_session_prompt_threshold == 2048
    assert reloaded.config.backend.sticky_max_sessions == 256
    assert reloaded.config.backend.split_mode == "row"
    assert reloaded.config.backend.no_context_shift is True
    assert reloaded.config.backend.jinja is True
    assert reloaded.config.backend.reasoning_format == "deepseek"
    assert reloaded.config.backend.prefill_strategy == "fixed"
    assert reloaded.config.backend.chunked_prefill_size == 8192
    assert reloaded.config.backend.fixed_chunked_prefill_size == 8192
    assert reloaded.config.backend.adaptive_short_prompt_threshold == 2048
    assert reloaded.config.backend.adaptive_long_context_chunk_size == 8192
    assert reloaded.config.backend.adaptive_repeat_prefix_chunk_size == 1024
    assert reloaded.config.backend.hicache_storage_backend == "file"
    assert (
        reloaded.config.backend.hicache_storage_backend_extra_config[
            "hicache_storage_pass_prefix_keys"
        ]
        is True
    )
    assert reloaded.config.backend.model_pool.max_loaded_models == 3
    assert reloaded.config.backend.model_pool.models == {}


def test_settings_manager_persists_llama_cpp_model_pool_registration(tmp_path: Path):
    manager = DGXSettingsManager(tmp_path)
    manager.ensure_llama_cpp_model_registration(
        LlamaCppModelRegistration(
            model_id="qwen35-4b-secondary",
            model_alias="qwen35-secondary",
            artifact_path="/models/Qwen3.5-4B-Q4_K_S.gguf",
            base_url="http://127.0.0.1:32121",
            gguf_variant="Q4_K_S",
            pinned=False,
            ttl_seconds=900,
            idle_unload_seconds=120,
            supports_vision=True,
            primary_service="ocr",
        ),
        profile=ModelProfile(
            model_id="qwen35-4b-secondary",
            model_alias="qwen35-secondary",
            supports_vision=True,
            primary_service="ocr",
        ),
    )

    reloaded = DGXSettingsManager(tmp_path)
    registration = reloaded.config.backend.model_pool.models["qwen35-4b-secondary"]

    assert reloaded.config.resolve_model_id("qwen35-secondary") == "qwen35-4b-secondary"
    assert registration.artifact_path == "/models/Qwen3.5-4B-Q4_K_S.gguf"
    assert registration.base_url == "http://127.0.0.1:32121"
    assert registration.gguf_variant == "Q4_K_S"
    assert registration.ttl_seconds == 900
    assert registration.idle_unload_seconds == 120
    assert registration.supports_vision is True
    assert registration.primary_service == "ocr"
    assert reloaded.config.models["qwen35-4b-secondary"].primary_service == "ocr"
    assert reloaded.config.public_models()[0]["capabilities"]["vision_chat"] is True
    assert reloaded.config.public_models()[0]["primary_service"] == "ocr"


def test_model_profile_infers_multimodal_capabilities_from_name():
    profile = ModelProfile(model_id="Qwen3-VL-8B-Instruct", model_alias="qwen3-vl")
    assert profile.supports_vision is True
    assert profile.supports_ocr is False

    ocr_profile = ModelProfile(model_id="GLM-OCR")
    assert ocr_profile.supports_vision is True
    assert ocr_profile.supports_ocr is True

    embedding_profile = ModelProfile(model_id="text-embedding-nomic-embed-text-v1.5")
    assert embedding_profile.supports_embeddings is True
    assert embedding_profile.supports_rerank is False
    assert embedding_profile.primary_service == "embeddings"

    rerank_profile = ModelProfile(model_id="Qwen3-Reranker-0.6B")
    assert rerank_profile.supports_rerank is True
    assert rerank_profile.primary_service == "rerank"


def test_manifest_store_round_trip(tmp_path: Path):
    store = PersistentManifestStore(tmp_path / "cache")
    record = StoredBlockRecord(
        block_hash="ab" * 32,
        model_id="qwen35-35b",
        token_count=256,
        tier="ssd",
        serializer="trt-page-v1",
        payload_path="payloads/ab/block.bin",
    )

    store.put(record)
    loaded = store.get(record.block_hash)

    assert loaded is not None
    assert loaded.model_id == "qwen35-35b"
    assert store.stats()["records"] == 1


def test_cli_serve_persists_hicache_flags(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--base-path",
            str(tmp_path),
            "--model-id",
            "Qwen/Qwen3.5-4B",
            "--model-alias",
            "qwen35-4b",
            "--quant-mode",
            "awq_int4",
            "--model-source",
            "hf",
            "--artifact-path",
            "/models/Qwen3.5-4B-AWQ",
            "--mamba-ssm-dtype",
            "float16",
            "--disable-cuda-graph",
            "--prefill-strategy",
            "adaptive",
            "--fixed-chunked-prefill-size",
            "4096",
            "--adaptive-short-prompt-threshold",
            "1024",
            "--adaptive-long-context-chunk-size",
            "4096",
            "--adaptive-repeat-prefix-chunk-size",
            "1024",
            "--adaptive-backend-url",
            "http://127.0.0.1:31001",
            "--enable-hierarchical-cache",
            "--hicache-size-gb",
            "8",
            "--hicache-pass-prefix-keys",
        ]
    )
    args.func(args)

    reloaded = DGXSettingsManager(tmp_path)
    assert reloaded.config.backend.quant_mode == "awq_int4"
    assert reloaded.config.backend.model_source == "hf"
    assert reloaded.config.backend.artifact_path == "/models/Qwen3.5-4B-AWQ"
    assert reloaded.config.backend.mamba_ssm_dtype == "float16"
    assert reloaded.config.backend.disable_cuda_graph is True
    assert reloaded.config.backend.prefill_strategy == "adaptive"
    assert reloaded.config.backend.fixed_chunked_prefill_size == 4096
    assert reloaded.config.backend.chunked_prefill_size == 4096
    assert reloaded.config.backend.adaptive_short_prompt_threshold == 1024
    assert reloaded.config.backend.adaptive_long_context_chunk_size == 4096
    assert reloaded.config.backend.adaptive_repeat_prefix_chunk_size == 1024
    assert reloaded.config.backend.adaptive_backend_base_url == "http://127.0.0.1:31001"
    assert reloaded.config.backend.enable_hierarchical_cache is True
    assert reloaded.config.backend.hicache_size == 8
    assert (
        reloaded.config.backend.hicache_storage_backend_extra_config[
            "hicache_storage_pass_prefix_keys"
        ]
        is True
    )


def test_settings_manager_migrates_legacy_chunked_prefill_size(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        """
{
  "backend": {
    "chunked_prefill_size": 2048
  }
}
""".strip(),
        encoding="utf-8",
    )

    reloaded = DGXSettingsManager(tmp_path)

    assert reloaded.config.backend.chunked_prefill_size == 2048
    assert reloaded.config.backend.fixed_chunked_prefill_size == 2048
    assert reloaded.config.backend.adaptive_long_context_chunk_size == 2048


def test_backend_config_gguf_quant_mode_forces_gguf_source(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        """
{
  "backend": {
    "quant_mode": "gguf_experimental",
    "model_source": "hf"
  }
}
""".strip(),
        encoding="utf-8",
    )

    reloaded = DGXSettingsManager(tmp_path)

    assert reloaded.config.backend.quant_mode == "gguf_experimental"
    assert reloaded.config.backend.model_source == "gguf"


def test_backend_config_applies_recommended_llama_cpp_gguf_defaults(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        """
{
  "backend": {
    "kind": "llama_cpp",
    "quant_mode": "gguf_experimental"
  }
}
""".strip(),
        encoding="utf-8",
    )

    reloaded = DGXSettingsManager(tmp_path)

    assert reloaded.config.backend.model_source == "gguf"
    assert reloaded.config.backend.gguf_variant == RECOMMENDED_LLAMA_CPP_GGUF_VARIANT
    assert reloaded.config.backend.model_repo_id == RECOMMENDED_LLAMA_CPP_GGUF_MODEL_REPO_ID


def test_cli_serve_persists_llama_cpp_fields(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--base-path",
            str(tmp_path),
            "--backend-kind",
            "llama_cpp",
            "--model-id",
            "lmstudio-community/Qwen3.5-4B-GGUF:Q4_K_M",
            "--model-alias",
            "qwen35-4b-q4km",
            "--quant-mode",
            "gguf_experimental",
            "--model-source",
            "gguf",
            "--artifact-path",
            "/models/Qwen3.5-4B-Q4_K_M.gguf",
            "--gguf-variant",
            "Q4_K_M",
            "--launcher-binary",
            "/opt/llama.cpp/bin/llama-server",
            "--launcher-cmd",
            "/opt/llama.cpp/bin/llama-server --model /models/Qwen3.5-4B-Q4_K_M.gguf --port 30000",
            "--ctx-size",
            "16384",
            "--parallel-slots",
            "2",
            "--n-gpu-layers",
            "999",
            "--flash-attn",
            "--batch-size",
            "4096",
            "--ubatch-size",
            "1024",
            "--cache-ram-mib",
            "16384",
            "--cache-reuse",
            "256",
            "--checkpoint-every-n-tokens",
            "1024",
            "--ctx-checkpoints",
            "64",
            "--slot-prompt-similarity",
            "0.25",
            "--enable-runtime-metrics",
            "--enable-session-stickiness",
            "--sticky-session-prompt-threshold",
            "3072",
            "--sticky-max-sessions",
            "128",
            "--split-mode",
            "row",
            "--no-context-shift",
            "--jinja",
            "--reasoning-format",
            "deepseek",
        ]
    )
    args.func(args)

    reloaded = DGXSettingsManager(tmp_path)
    assert reloaded.config.backend.kind == "llama_cpp"
    assert reloaded.config.backend.quant_mode == "gguf_experimental"
    assert reloaded.config.backend.model_source == "gguf"
    assert reloaded.config.backend.artifact_path == "/models/Qwen3.5-4B-Q4_K_M.gguf"
    assert reloaded.config.backend.gguf_variant == "Q4_K_M"
    assert reloaded.config.backend.launcher_binary == "/opt/llama.cpp/bin/llama-server"
    assert (
        reloaded.config.backend.launcher_cmd
        == "/opt/llama.cpp/bin/llama-server --model /models/Qwen3.5-4B-Q4_K_M.gguf --port 30000"
    )
    assert reloaded.config.backend.ctx_size == 16384
    assert reloaded.config.backend.parallel_slots == 2
    assert reloaded.config.backend.n_gpu_layers == 999
    assert reloaded.config.backend.flash_attn is True
    assert reloaded.config.backend.batch_size == 4096
    assert reloaded.config.backend.ubatch_size == 1024
    assert reloaded.config.backend.cache_ram_mib == 16384
    assert reloaded.config.backend.cache_reuse == 256
    assert reloaded.config.backend.checkpoint_every_n_tokens == 1024
    assert reloaded.config.backend.ctx_checkpoints == 64
    assert reloaded.config.backend.slot_prompt_similarity == 0.25
    assert reloaded.config.backend.enable_runtime_metrics is True
    assert reloaded.config.backend.enable_session_stickiness is True
    assert reloaded.config.backend.sticky_session_prompt_threshold == 3072
    assert reloaded.config.backend.sticky_max_sessions == 128
    assert reloaded.config.backend.split_mode == "row"
    assert reloaded.config.backend.no_context_shift is True
    assert reloaded.config.backend.jinja is True
    assert reloaded.config.backend.reasoning_format == "deepseek"


def test_cli_serve_applies_llama_cpp_single_session_preset(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(cli, "create_app", lambda **kwargs: object())
    monkeypatch.setattr(cli.uvicorn, "run", lambda *args, **kwargs: None)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "serve",
            "--base-path",
            str(tmp_path),
            "--backend-kind",
            "llama_cpp",
            "--model-id",
            "lmstudio-community/Qwen3.5-4B-GGUF:Q4_K_S",
            "--model-alias",
            "qwen35-4b",
            "--quant-mode",
            "gguf_experimental",
            "--model-source",
            "gguf",
            "--artifact-path",
            "/models/Qwen3.5-4B-Q4_K_S.gguf",
            "--serving-preset",
            "single_session_low_latency",
        ]
    )
    args.func(args)

    reloaded = DGXSettingsManager(tmp_path)
    assert reloaded.config.backend.serving_preset == "single_session_low_latency"
    assert reloaded.config.backend.ctx_size == 32768
    assert reloaded.config.backend.parallel_slots == 1
    assert reloaded.config.backend.batch_size == 8192
    assert reloaded.config.backend.ubatch_size == 2048
    assert reloaded.config.backend.cache_ram_mib == 16384
    assert reloaded.config.backend.cache_reuse == 256
    assert reloaded.config.backend.checkpoint_every_n_tokens == 1024
    assert reloaded.config.backend.ctx_checkpoints == 64
    assert reloaded.config.backend.enable_runtime_metrics is True


def test_settings_manager_auto_upgrades_legacy_llama_cpp_defaults(tmp_path: Path):
    (tmp_path / "settings.json").write_text(
        """
{
  "backend": {
    "kind": "llama_cpp",
    "quant_mode": "gguf_experimental",
    "model_source": "gguf",
    "artifact_path": "/models/Qwen3.5-4B-Q4_K_S.gguf"
  }
}
""".strip(),
        encoding="utf-8",
    )

    reloaded = DGXSettingsManager(tmp_path)

    assert reloaded.config.backend.serving_preset == "single_session_low_latency"
    assert reloaded.config.backend.ctx_size == 32768
    assert reloaded.config.backend.parallel_slots == 1
    assert reloaded.config.backend.batch_size == 8192
    assert reloaded.config.backend.ubatch_size == 2048
    assert reloaded.config.backend.cache_ram_mib == 16384
    assert reloaded.config.backend.cache_reuse == 256
    assert reloaded.config.backend.checkpoint_every_n_tokens == 1024
    assert reloaded.config.backend.ctx_checkpoints == 64
