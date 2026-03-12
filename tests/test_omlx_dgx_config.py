# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import omlx_dgx.cli as cli
from omlx_dgx.config import DGXSettingsManager, ModelProfile
from omlx_dgx.tiered_kv import PersistentManifestStore, StoredBlockRecord


def test_settings_manager_persists_default_model(tmp_path: Path):
    manager = DGXSettingsManager(tmp_path)
    manager.ensure_model(ModelProfile(model_id="qwen35-35b", model_alias="qwen35", is_default=True))

    reloaded = DGXSettingsManager(tmp_path)
    assert reloaded.config.resolve_model_id("qwen35") == "qwen35-35b"
    assert reloaded.config.resolve_model_id(None) == "qwen35-35b"
    assert reloaded.config.backend.kind == "sglang"
    assert reloaded.config.backend.model_repo_id == "Qwen/Qwen3.5-35B-A3B"
    assert reloaded.config.backend.attention_backend == "triton"
    assert reloaded.config.backend.chat_template == ""
    assert reloaded.config.backend.chunked_prefill_size == 8192
    assert reloaded.config.backend.hicache_storage_backend == "file"
    assert (
        reloaded.config.backend.hicache_storage_backend_extra_config[
            "hicache_storage_pass_prefix_keys"
        ]
        is True
    )


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
            "--chunked-prefill-size",
            "1024",
            "--enable-hierarchical-cache",
            "--hicache-size-gb",
            "8",
            "--hicache-pass-prefix-keys",
        ]
    )
    args.func(args)

    reloaded = DGXSettingsManager(tmp_path)
    assert reloaded.config.backend.chunked_prefill_size == 1024
    assert reloaded.config.backend.enable_hierarchical_cache is True
    assert reloaded.config.backend.hicache_size == 8
    assert (
        reloaded.config.backend.hicache_storage_backend_extra_config[
            "hicache_storage_pass_prefix_keys"
        ]
        is True
    )
