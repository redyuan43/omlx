# SPDX-License-Identifier: Apache-2.0
"""Tiered KV metadata and persistent cold-store manifest handling."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional


class CacheTier(str, Enum):
    GPU = "gpu"
    HOST = "host"
    SSD = "ssd"


@dataclass
class RetentionPolicy:
    """Retention hints surfaced to the backend scheduler/runtime."""

    priority: int = 50
    pinned: bool = False
    ttl_seconds: Optional[int] = None


@dataclass
class StoredBlockRecord:
    """Serializable metadata for one persisted KV block."""

    block_hash: str
    model_id: str
    token_count: int
    tier: str
    serializer: str
    payload_path: str
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    parent_hash: Optional[str] = None
    extra_keys: List[str] = field(default_factory=list)
    retention_priority: int = 50
    checksum: Optional[str] = None


@dataclass
class KVRestorePlan:
    """Plan for restoring a list of block hashes from the tiered cache."""

    model_id: str
    hits: List[StoredBlockRecord] = field(default_factory=list)
    misses: List[str] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        total = len(self.hits) + len(self.misses)
        if total == 0:
            return 0.0
        return len(self.hits) / total


class PersistentManifestStore:
    """Durable manifest store for SSD-cold KV block metadata."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index_root = self.root / "index"
        self.payload_root = self.root / "payloads"
        self.wal_path = self.root / "manifest.wal"
        self.index_root.mkdir(parents=True, exist_ok=True)
        self.payload_root.mkdir(parents=True, exist_ok=True)
        self.wal_path.touch(exist_ok=True)

    def _record_path(self, block_hash: str) -> Path:
        prefix = block_hash[:2]
        path = self.index_root / prefix
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{block_hash}.json"

    def put(self, record: StoredBlockRecord) -> StoredBlockRecord:
        path = self._record_path(record.block_hash)
        path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
        with self.wal_path.open("a", encoding="utf-8") as wal:
            wal.write(json.dumps({"op": "put", "record": asdict(record)}) + "\n")
        return record

    def get(self, block_hash: str) -> Optional[StoredBlockRecord]:
        path = self._record_path(block_hash)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        record = StoredBlockRecord(**data)
        record.last_access = time.time()
        self.put(record)
        return record

    def delete(self, block_hash: str) -> bool:
        path = self._record_path(block_hash)
        if not path.exists():
            return False
        path.unlink()
        with self.wal_path.open("a", encoding="utf-8") as wal:
            wal.write(json.dumps({"op": "delete", "block_hash": block_hash}) + "\n")
        return True

    def list_records(self, *, model_id: Optional[str] = None, limit: int = 100) -> List[StoredBlockRecord]:
        records: List[StoredBlockRecord] = []
        for file_path in sorted(self.index_root.glob("*/*.json")):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            record = StoredBlockRecord(**data)
            if model_id is not None and record.model_id != model_id:
                continue
            records.append(record)
            if len(records) >= limit:
                break
        records.sort(key=lambda item: item.last_access, reverse=True)
        return records

    def build_restore_plan(self, model_id: str, block_hashes: List[str]) -> KVRestorePlan:
        plan = KVRestorePlan(model_id=model_id)
        for block_hash in block_hashes:
            record = self.get(block_hash)
            if record is not None and record.model_id == model_id:
                plan.hits.append(record)
            else:
                plan.misses.append(block_hash)
        return plan

    def stats(self) -> Dict[str, int]:
        count = 0
        bytes_on_disk = 0
        for file_path in self.index_root.glob("*/*.json"):
            count += 1
            bytes_on_disk += file_path.stat().st_size
        return {
            "records": count,
            "metadata_bytes": bytes_on_disk,
            "wal_bytes": self.wal_path.stat().st_size if self.wal_path.exists() else 0,
        }
