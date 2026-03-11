# SPDX-License-Identifier: Apache-2.0
"""Backend-agnostic block metadata and prefix-cache helpers for DGX runtimes."""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, NewType, Optional, Tuple

BlockHash = NewType("BlockHash", bytes)


def compute_block_hash(
    parent_hash: Optional[BlockHash],
    token_ids: List[int],
    *,
    model_id: str,
    extra_keys: Optional[Tuple[str, ...]] = None,
) -> BlockHash:
    """Create an oMLX-style chain hash for a token block."""
    hasher = hashlib.sha256()
    hasher.update(model_id.encode("utf-8"))
    hasher.update(parent_hash or b"omlx-dgx-root")
    hasher.update(",".join(str(token) for token in token_ids).encode("utf-8"))
    if extra_keys:
        hasher.update("|".join(extra_keys).encode("utf-8"))
    return BlockHash(hasher.digest())


@dataclass
class CacheBlock:
    """Physical cache block metadata shared across all tiers."""

    block_id: int
    block_hash: Optional[BlockHash] = None
    token_count: int = 0
    ref_count: int = 0
    tier: str = "gpu"
    last_access: float = field(default_factory=time.time)
    prev_free_block: Optional["CacheBlock"] = None
    next_free_block: Optional["CacheBlock"] = None

    def touch(self) -> None:
        self.last_access = time.time()

    def is_shared(self) -> bool:
        return self.ref_count > 1


class FreeBlockQueue:
    """O(1) doubly-linked free list following the oMLX/vLLM design."""

    def __init__(self, blocks: List[CacheBlock]) -> None:
        self.num_free_blocks = len(blocks)
        self.fake_head = CacheBlock(block_id=-1)
        self.fake_tail = CacheBlock(block_id=-2)

        if not blocks:
            self.fake_head.next_free_block = self.fake_tail
            self.fake_tail.prev_free_block = self.fake_head
            return

        prev = self.fake_head
        for block in blocks:
            prev.next_free_block = block
            block.prev_free_block = prev
            prev = block
        prev.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = prev

    def popleft(self) -> CacheBlock:
        if self.fake_head.next_free_block is self.fake_tail:
            raise ValueError("No free blocks available")
        block = self.fake_head.next_free_block
        assert block is not None
        self.remove(block)
        return block

    def append(self, block: CacheBlock) -> None:
        tail_prev = self.fake_tail.prev_free_block
        assert tail_prev is not None
        tail_prev.next_free_block = block
        block.prev_free_block = tail_prev
        block.next_free_block = self.fake_tail
        self.fake_tail.prev_free_block = block
        self.num_free_blocks += 1

    def remove(self, block: CacheBlock) -> None:
        prev_block = block.prev_free_block
        next_block = block.next_free_block
        if prev_block is None or next_block is None:
            raise RuntimeError(f"Block {block.block_id} is not in free queue")
        prev_block.next_free_block = next_block
        next_block.prev_free_block = prev_block
        block.prev_free_block = None
        block.next_free_block = None
        self.num_free_blocks -= 1

    def get_all_free_blocks(self) -> List[CacheBlock]:
        blocks: List[CacheBlock] = []
        cursor = self.fake_head.next_free_block
        while cursor is not None and cursor is not self.fake_tail:
            blocks.append(cursor)
            cursor = cursor.next_free_block
        return blocks


class BlockLedger:
    """Content-addressed block metadata store with LRU lookup."""

    def __init__(self) -> None:
        self._by_hash: "OrderedDict[bytes, CacheBlock]" = OrderedDict()
        self._by_id: Dict[int, CacheBlock] = {}

    def register(self, block: CacheBlock) -> None:
        self._by_id[block.block_id] = block
        if block.block_hash is not None:
            self._by_hash[bytes(block.block_hash)] = block
            self._by_hash.move_to_end(bytes(block.block_hash))

    def get_by_hash(self, block_hash: BlockHash) -> Optional[CacheBlock]:
        key = bytes(block_hash)
        block = self._by_hash.get(key)
        if block is not None:
            block.touch()
            self._by_hash.move_to_end(key)
        return block

    def evict_lru(self) -> Optional[CacheBlock]:
        if not self._by_hash:
            return None
        _, block = self._by_hash.popitem(last=False)
        self._by_id.pop(block.block_id, None)
        return block

    def snapshot(self) -> List[dict]:
        return [
            {
                "block_id": block.block_id,
                "tier": block.tier,
                "token_count": block.token_count,
                "ref_count": block.ref_count,
                "last_access": block.last_access,
                "block_hash": block.block_hash.hex() if block.block_hash else None,
            }
            for block in self._by_hash.values()
        ]
