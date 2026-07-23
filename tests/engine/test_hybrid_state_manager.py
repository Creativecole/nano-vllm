from types import SimpleNamespace

import pytest
import torch

from nanovllm.engine.layer_state import (
    DeltaNetStateSpec,
    HybridStateManager,
    delta_state_bytes_per_sequence,
)
from nanovllm.engine.model_runner import ModelRunner


def make_spec(layer_idx=0):
    return DeltaNetStateSpec(
        layer_idx=layer_idx,
        layer_type="linear_attention",
        conv_dim=16,
        conv_width=4,
        num_value_heads=4,
        key_head_dim=8,
        value_head_dim=8,
        conv_dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_allocate_gather_commit_and_free(batch_size):
    manager = HybridStateManager([make_spec()], capacity=4, device="cpu")
    seq_ids = list(range(10, 10 + batch_size))
    manager.allocate(seq_ids)
    states = manager.gather(seq_ids)
    states[0].conv_state.fill_(3)
    states[0].recurrent_state.fill_(7)
    manager.commit(states)

    gathered = manager.gather(list(reversed(seq_ids)))[0]
    assert torch.all(gathered.conv_state == 3)
    assert torch.all(gathered.recurrent_state == 7)
    manager.free(seq_ids)
    assert manager.allocated_count == 0
    assert manager.free_count == 4


def test_reorder_preserves_request_identity():
    manager = HybridStateManager([make_spec(3), make_spec(7)], capacity=4, device="cpu")
    seq_ids = [101, 202, 303]
    manager.allocate(seq_ids)
    states = manager.gather(seq_ids)
    for row, seq_id in enumerate(seq_ids):
        states[3].recurrent_state[row].fill_(seq_id)
        states[7].conv_state[row].fill_(seq_id)
    manager.commit(states)

    reordered_ids = [303, 101, 202]
    reordered = manager.reorder(reordered_ids)
    for row, seq_id in enumerate(reordered_ids):
        assert torch.all(reordered[3].recurrent_state[row] == seq_id)
        assert torch.all(reordered[7].conv_state[row] == seq_id)


def test_reused_slot_is_zeroed_and_does_not_leak_state():
    manager = HybridStateManager([make_spec()], capacity=1, device="cpu")
    manager.allocate([1])
    state = manager.gather([1])
    state[0].conv_state.fill_(9)
    state[0].recurrent_state.fill_(9)
    manager.commit(state)
    manager.free([1])
    manager.allocate([2])
    reused = manager.gather([2])[0]
    assert torch.count_nonzero(reused.conv_state) == 0
    assert torch.count_nonzero(reused.recurrent_state) == 0


def test_statecopy_mode_reuses_free_list_without_compaction():
    manager = HybridStateManager(
        [make_spec()],
        capacity=3,
        device="cpu",
        compact_slots=False,
    )
    manager.allocate([10, 20, 30])
    manager.free([20])

    assert manager.slot_for(10) == 0
    assert manager.slot_for(30) == 2
    manager.allocate([40])
    assert manager.slot_for(40) == 1
    assert manager.get_runtime_stats()["compaction_calls"] == 0


def test_pool_exhaustion_is_explicit():
    manager = HybridStateManager([make_spec()], capacity=1, device="cpu")
    manager.allocate([1])
    with pytest.raises(RuntimeError, match="state pool exhausted"):
        manager.allocate([2])


def test_state_bytes_are_derived_from_spec():
    spec = make_spec()
    expected = 16 * 4 * torch.bfloat16.itemsize
    expected += 4 * 8 * 8 * torch.float32.itemsize
    assert delta_state_bytes_per_sequence([spec]) == expected


def test_state_manager_diagnostics_report_batched_state_movement():
    specs = [make_spec(3), make_spec(7)]
    manager = HybridStateManager(specs, capacity=4, device="cpu")
    seq_ids = [10, 20, 30, 40]
    manager.allocate(seq_ids)
    manager.set_diagnostics(True)

    states = manager.gather(seq_ids)
    manager.commit(states)

    stats = manager.get_diagnostics()
    expected_bytes = len(seq_ids) * delta_state_bytes_per_sequence(specs)
    assert stats == {
        "gather_calls": 1,
        "gather_layer_ops": 2,
        "gather_bytes": expected_bytes,
        "commit_calls": 1,
        "commit_layer_ops": 2,
        "commit_bytes": expected_bytes,
        "slot_id_upload_bytes": len(seq_ids) * torch.long.itemsize,
        "batch_sizes": [4],
        "resident_view_calls": 0,
        "resident_layer_views": 0,
        "resident_rows": 0,
        "estimated_state_copy_bytes_avoided": 0,
        "resident_commit_skips": 0,
        "compaction_calls": 0,
        "compaction_layer_ops": 0,
        "compaction_bytes": 0,
    }


def test_resident_view_updates_pool_without_gather_or_commit():
    specs = [make_spec(3), make_spec(7)]
    manager = HybridStateManager(specs, capacity=4, device="cpu")
    seq_ids = [10, 20, 30]
    manager.allocate(seq_ids)

    ordered = manager.resident_order(list(reversed(seq_ids)))
    assert ordered == seq_ids
    states = manager.resident(ordered)
    for row, seq_id in enumerate(ordered):
        states[3].recurrent_state[row].fill_(seq_id)
        states[7].conv_state[row].fill_(seq_id)

    assert manager.commit(states) is False
    assert torch.all(manager.pools[3].recurrent_state[1] == 20)
    assert torch.all(manager.pools[7].conv_state[2] == 30)
    assert manager.get_runtime_stats() == {
        "allocation_calls": 1,
        "allocation_rows": 3,
        "allocation_zero_ops": 2 * len(specs),
        "resident_view_calls": 1,
        "resident_layer_views": 2,
        "resident_rows": 3,
        "estimated_state_copy_bytes_avoided": (
            2 * len(seq_ids) * delta_state_bytes_per_sequence(specs)
        ),
        "fallback_gather_calls": 0,
        "fallback_commit_calls": 0,
        "resident_commit_skips": 1,
        "compaction_calls": 0,
        "compaction_layer_ops": 0,
        "compaction_bytes": 0,
    }


def test_free_compacts_resident_slots_and_preserves_moved_request():
    specs = [make_spec(3), make_spec(7)]
    manager = HybridStateManager(specs, capacity=4, device="cpu")
    manager.allocate([10, 20, 30])
    states = manager.resident([10, 20, 30])
    for row, seq_id in enumerate([10, 20, 30]):
        states[3].recurrent_state[row].fill_(seq_id)
        states[7].conv_state[row].fill_(seq_id)

    manager.free([20])

    assert manager.slot_for(10) == 0
    assert manager.slot_for(30) == 1
    assert manager.resident_order([30, 10]) == [10, 30]
    compacted = manager.resident([10, 30])
    assert torch.all(compacted[3].recurrent_state[1] == 30)
    assert torch.all(compacted[7].conv_state[1] == 30)

    manager.allocate([40])
    assert manager.slot_for(40) == 2
    reused = manager.resident([40])[3]
    assert torch.count_nonzero(reused.conv_state) == 0
    assert torch.count_nonzero(reused.recurrent_state) == 0

    runtime = manager.get_runtime_stats()
    assert runtime["compaction_calls"] == 1
    assert runtime["compaction_layer_ops"] == len(specs)
    assert runtime["compaction_bytes"] == delta_state_bytes_per_sequence(specs)


def test_compact_allocation_batches_state_zeroing_by_layer():
    specs = [make_spec(3), make_spec(7)]
    manager = HybridStateManager(specs, capacity=4, device="cpu")

    manager.allocate([10, 20, 30])

    runtime = manager.get_runtime_stats()
    assert runtime["allocation_calls"] == 1
    assert runtime["allocation_rows"] == 3
    assert runtime["allocation_zero_ops"] == 2 * len(specs)


def test_noncontiguous_subset_requires_gather_fallback():
    manager = HybridStateManager([make_spec()], capacity=4, device="cpu")
    manager.allocate([10, 20, 30])

    assert manager.resident_order([10, 30]) is None
    states = manager.gather([10, 30])
    states[0].recurrent_state[1].fill_(30)
    assert manager.commit(states) is True
    assert torch.all(manager.pools[0].recurrent_state[2] == 30)


def test_model_runner_orders_resident_batch_and_restores_outputs():
    manager = HybridStateManager([make_spec()], capacity=4, device="cpu")
    manager.allocate([10, 20])
    runner = ModelRunner.__new__(ModelRunner)
    runner.hybrid_state_manager = manager
    runner.config = SimpleNamespace(resident_deltanet_state=True)
    runner.execution_stats = {
        "state_resident_view_calls": 0,
        "state_gather_calls": 0,
        "state_commit_calls": 0,
        "state_commit_skipped_calls": 0,
    }
    requested = [
        SimpleNamespace(seq_id=20),
        SimpleNamespace(seq_id=10),
    ]

    ordered, states, is_resident = runner._prepare_hybrid_state_batch(
        requested,
        allow_reorder=True,
    )

    assert [seq.seq_id for seq in ordered] == [10, 20]
    assert is_resident
    states[0].recurrent_state[0].fill_(10)
    states[0].recurrent_state[1].fill_(20)
    runner._finish_hybrid_state_batch(states, is_resident)
    assert runner.execution_stats["state_resident_view_calls"] == 1
    assert runner.execution_stats["state_commit_skipped_calls"] == 1
    assert runner.execution_stats["state_gather_calls"] == 0
    assert runner.execution_stats["state_commit_calls"] == 0
    assert runner._restore_request_order([10, 20], ordered, requested) == [20, 10]
    restored = runner._restore_request_order(
        torch.tensor([[10], [20]]),
        ordered,
        requested,
    )
    assert torch.equal(restored, torch.tensor([[20], [10]]))


def test_model_runner_can_disable_resident_state_for_ab_validation():
    manager = HybridStateManager([make_spec()], capacity=4, device="cpu")
    runner = ModelRunner.__new__(ModelRunner)
    runner.hybrid_state_manager = manager
    runner.config = SimpleNamespace(resident_deltanet_state=False)
    runner.execution_stats = {
        "state_resident_view_calls": 0,
        "state_gather_calls": 0,
        "state_commit_calls": 0,
        "state_commit_skipped_calls": 0,
    }
    requested = [
        SimpleNamespace(seq_id=10),
        SimpleNamespace(seq_id=20),
    ]

    ordered, states, is_resident = runner._prepare_hybrid_state_batch(
        requested,
        allow_reorder=True,
    )

    assert ordered == requested
    assert not is_resident
    states[0].recurrent_state[0].fill_(10)
    states[0].recurrent_state[1].fill_(20)
    runner._finish_hybrid_state_batch(states, is_resident)
    assert runner.execution_stats["state_resident_view_calls"] == 0
    assert runner.execution_stats["state_gather_calls"] == 1
    assert runner.execution_stats["state_commit_calls"] == 1
    runtime = manager.get_runtime_stats()
    assert runtime["fallback_gather_calls"] == 1
    assert runtime["fallback_commit_calls"] == 1
