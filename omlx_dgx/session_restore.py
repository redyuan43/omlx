# SPDX-License-Identifier: Apache-2.0
"""Persistent metadata snapshots for DGX session restore."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def snapshot_key(model_id: str, conversation_id: str) -> str:
    digest = hashlib.sha256(
        f"{model_id}\0{conversation_id}".encode("utf-8")
    ).hexdigest()
    return digest[:32]


def conversation_key_hash(conversation_id: str) -> str:
    return hashlib.sha256(conversation_id.encode("utf-8")).hexdigest()[:12]


@dataclass
class SessionRestoreSnapshot:
    conversation_id: str
    model_id: str
    slot_id: int
    estimated_prompt_tokens: int
    prefix_digest: str
    request_shape_digest: str
    prompt_mode: str
    message_count: int
    slot_message_count: int
    save_filename: str
    state_payload: Dict[str, Any] = field(default_factory=dict)
    saved_at: float = field(default_factory=time.time)
    last_access_at: float = field(default_factory=time.time)
    save_ms: Optional[float] = None
    n_saved: int = 0
    n_written: int = 0
    restore_count: int = 0
    last_restore_at: Optional[float] = None
    last_restore_ms: Optional[float] = None
    last_restore_n_restored: int = 0
    last_restore_n_read: int = 0
    last_restore_status: str = ""
    last_restore_error: str = ""
    runtime_signature: str = ""

    @property
    def key(self) -> str:
        return snapshot_key(self.model_id, self.conversation_id)

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "conversation_key_hash": conversation_key_hash(self.conversation_id),
            "model_id": self.model_id,
            "slot_id": self.slot_id,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "prefix_digest": self.prefix_digest,
            "request_shape_digest": self.request_shape_digest,
            "prompt_mode": self.prompt_mode,
            "message_count": self.message_count,
            "slot_message_count": self.slot_message_count,
            "save_filename": self.save_filename,
            "saved_at": round(self.saved_at, 3),
            "last_access_at": round(self.last_access_at, 3),
            "save_ms": None if self.save_ms is None else round(self.save_ms, 3),
            "n_saved": self.n_saved,
            "n_written": self.n_written,
            "restore_count": self.restore_count,
            "last_restore_at": (
                None if self.last_restore_at is None else round(self.last_restore_at, 3)
            ),
            "last_restore_ms": (
                None if self.last_restore_ms is None else round(self.last_restore_ms, 3)
            ),
            "last_restore_n_restored": self.last_restore_n_restored,
            "last_restore_n_read": self.last_restore_n_read,
            "last_restore_status": self.last_restore_status,
            "last_restore_error": self.last_restore_error,
            "runtime_signature": self.runtime_signature,
        }


class SessionRestoreStore:
    """Durable metadata store for managed llama.cpp slot save/restore."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.index_root = self.root / "index"
        self.index_root.mkdir(parents=True, exist_ok=True)

    def _record_path(self, key: str) -> Path:
        prefix = key[:2]
        path = self.index_root / prefix
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{key}.json"

    def put(self, snapshot: SessionRestoreSnapshot) -> SessionRestoreSnapshot:
        path = self._record_path(snapshot.key)
        path.write_text(
            json.dumps(asdict(snapshot), indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        return snapshot

    def get(
        self,
        *,
        conversation_id: str,
        model_id: str,
        touch: bool = True,
    ) -> Optional[SessionRestoreSnapshot]:
        key = snapshot_key(model_id, conversation_id)
        path = self._record_path(key)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        snapshot = SessionRestoreSnapshot(**data)
        if touch:
            snapshot.last_access_at = time.time()
            self.put(snapshot)
        return snapshot

    def delete(self, *, conversation_id: str, model_id: str) -> bool:
        key = snapshot_key(model_id, conversation_id)
        path = self._record_path(key)
        if not path.exists():
            return False
        path.unlink()
        return True

    def list_snapshots(
        self,
        *,
        model_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionRestoreSnapshot]:
        snapshots: List[SessionRestoreSnapshot] = []
        for file_path in sorted(self.index_root.glob("*/*.json")):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            snapshot = SessionRestoreSnapshot(**data)
            if model_id is not None and snapshot.model_id != model_id:
                continue
            snapshots.append(snapshot)
        snapshots.sort(key=lambda item: item.saved_at, reverse=True)
        return snapshots[:limit]

    def _iter_snapshots(
        self,
        *,
        model_id: Optional[str] = None,
    ):
        for file_path in self.index_root.glob("*/*.json"):
            data = json.loads(file_path.read_text(encoding="utf-8"))
            snapshot = SessionRestoreSnapshot(**data)
            if model_id is not None and snapshot.model_id != model_id:
                continue
            yield snapshot

    def find_candidates(
        self,
        *,
        model_id: str,
        request_shape_digest: Optional[str] = None,
        prompt_mode: Optional[str] = None,
        limit: int = 100,
    ) -> List[SessionRestoreSnapshot]:
        candidates: List[SessionRestoreSnapshot] = []
        for snapshot in self._iter_snapshots(model_id=model_id):
            if request_shape_digest and snapshot.request_shape_digest != request_shape_digest:
                continue
            if prompt_mode and snapshot.prompt_mode != prompt_mode:
                continue
            candidates.append(snapshot)
        candidates.sort(key=lambda item: item.last_access_at, reverse=True)
        return candidates[:limit]

    def find_exact_prefix_digest(
        self,
        *,
        model_id: str,
        prefix_digest: str,
        request_shape_digest: Optional[str] = None,
        prompt_mode: Optional[str] = None,
        exclude_conversation_id: Optional[str] = None,
        limit: int = 8,
    ) -> List[SessionRestoreSnapshot]:
        candidates: List[SessionRestoreSnapshot] = []
        for snapshot in self._iter_snapshots(model_id=model_id):
            if exclude_conversation_id and snapshot.conversation_id == exclude_conversation_id:
                continue
            if snapshot.prefix_digest != prefix_digest:
                continue
            if request_shape_digest and snapshot.request_shape_digest != request_shape_digest:
                continue
            if prompt_mode and snapshot.prompt_mode != prompt_mode:
                continue
            candidates.append(snapshot)
        candidates.sort(key=lambda item: item.last_access_at, reverse=True)
        return candidates[:limit]

    def find_prefix_candidates(
        self,
        *,
        model_id: str,
        request_shape_digest: Optional[str] = None,
        prompt_mode: Optional[str] = None,
        exclude_conversation_id: Optional[str] = None,
        max_estimated_prompt_tokens: Optional[int] = None,
        limit: int = 16,
    ) -> List[SessionRestoreSnapshot]:
        candidates: List[SessionRestoreSnapshot] = []
        for snapshot in self._iter_snapshots(model_id=model_id):
            if exclude_conversation_id and snapshot.conversation_id == exclude_conversation_id:
                continue
            if request_shape_digest and snapshot.request_shape_digest != request_shape_digest:
                continue
            if prompt_mode and snapshot.prompt_mode != prompt_mode:
                continue
            if (
                max_estimated_prompt_tokens is not None
                and snapshot.estimated_prompt_tokens > max_estimated_prompt_tokens
            ):
                continue
            candidates.append(snapshot)
        candidates.sort(key=lambda item: item.last_access_at, reverse=True)
        return candidates[:limit]

    def stats(self) -> Dict[str, int]:
        count = 0
        bytes_on_disk = 0
        for file_path in self.index_root.glob("*/*.json"):
            count += 1
            bytes_on_disk += file_path.stat().st_size
        return {
            "snapshots": count,
            "metadata_bytes": bytes_on_disk,
        }
