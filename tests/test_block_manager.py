from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_sequence(token_ids):
    return Sequence(token_ids, SamplingParams(max_tokens=1))


def test_cache_stats_track_block_usage():
    old_block_size = Sequence.block_size
    Sequence.block_size = 2
    try:
        manager = BlockManager(num_blocks=4, block_size=2)
        seq = make_sequence([1, 2, 3])

        assert manager.can_allocate(seq) == 0
        manager.allocate(seq, num_cached_blocks=0)
        stats = manager.get_cache_stats()
        assert stats["total_blocks"] == 4
        assert stats["used_blocks"] == 2
        assert stats["free_blocks"] == 2
        assert stats["max_used_blocks"] == 2
        assert stats["block_utilization"] == 0.5

        manager.deallocate(seq)
        stats = manager.get_cache_stats()
        assert stats["used_blocks"] == 0
        assert stats["free_blocks"] == 4
        assert stats["max_used_blocks"] == 2
    finally:
        Sequence.block_size = old_block_size


def test_prefix_cache_hit_rate_counts_full_block_reuse():
    old_block_size = Sequence.block_size
    Sequence.block_size = 2
    try:
        manager = BlockManager(num_blocks=4, block_size=2)
        first = make_sequence([10, 11, 12])
        assert manager.can_allocate(first) == 0
        manager.allocate(first, num_cached_blocks=0)
        first.num_scheduled_tokens = first.num_tokens
        manager.hash_blocks(first)
        manager.deallocate(first)

        second = make_sequence([10, 11, 99])
        assert manager.can_allocate(second) == 1
        stats = manager.get_cache_stats()
        assert stats["prefix_cache_hits"] == 1
        assert stats["prefix_cache_misses"] == 1
        assert stats["prefix_cache_hit_rate"] == 0.5
    finally:
        Sequence.block_size = old_block_size
