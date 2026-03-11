# SPDX-License-Identifier: Apache-2.0

from omlx_dgx.cache_core import BlockLedger, CacheBlock, FreeBlockQueue, compute_block_hash


def test_compute_block_hash_is_chain_sensitive():
    first = compute_block_hash(None, [1, 2, 3], model_id="qwen35")
    second = compute_block_hash(first, [4, 5, 6], model_id="qwen35")
    different_parent = compute_block_hash(None, [4, 5, 6], model_id="qwen35")

    assert first != second
    assert second != different_parent


def test_free_block_queue_round_trip():
    blocks = [CacheBlock(block_id=i) for i in range(3)]
    queue = FreeBlockQueue(blocks)

    popped = queue.popleft()
    assert popped.block_id == 0
    assert queue.num_free_blocks == 2

    queue.append(popped)
    assert queue.num_free_blocks == 3
    assert [block.block_id for block in queue.get_all_free_blocks()] == [1, 2, 0]


def test_block_ledger_tracks_lru_order():
    ledger = BlockLedger()
    first_hash = compute_block_hash(None, [1], model_id="qwen35")
    second_hash = compute_block_hash(first_hash, [2], model_id="qwen35")

    first = CacheBlock(block_id=1, block_hash=first_hash)
    second = CacheBlock(block_id=2, block_hash=second_hash)
    ledger.register(first)
    ledger.register(second)

    assert ledger.get_by_hash(first_hash) is first
    evicted = ledger.evict_lru()
    assert evicted is second
