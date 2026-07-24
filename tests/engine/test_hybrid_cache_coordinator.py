import torch

from nanovllm.engine.cache_coordinator import HybridCacheCoordinator
from nanovllm.engine.layer_state import DeltaNetStateSpec, PagedKVStateSpec


def test_hybrid_cache_coordinator_owns_both_state_kinds():
    specs = [
        DeltaNetStateSpec(
            layer_idx=0,
            layer_type="linear_attention",
            conv_dim=16,
            conv_width=4,
            num_value_heads=4,
            key_head_dim=8,
            value_head_dim=8,
            conv_dtype=torch.bfloat16,
        ),
        PagedKVStateSpec(
            layer_idx=1,
            layer_type="full_attention",
            num_kv_heads=2,
            head_dim=8,
            dtype=torch.bfloat16,
        ),
    ]
    coordinator = HybridCacheCoordinator(
        specs,
        state_capacity=4,
        num_kv_blocks=8,
        block_size=16,
        device="cpu",
        compact_delta_slots=True,
    )

    coordinator.allocate_requests([11, 22])
    stats = coordinator.get_stats()
    assert stats["allocated_state_slots"] == 2
    assert stats["full_attention_layers"] == 1
    assert coordinator.paged_kv_states[1].k_cache.shape == (8, 16, 2, 8)

    coordinator.free_requests([11, 22])
    assert coordinator.get_stats()["allocated_state_slots"] == 0
