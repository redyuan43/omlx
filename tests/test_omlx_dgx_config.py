# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

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
    assert reloaded.config.backend.hicache_storage_backend == "file"


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
