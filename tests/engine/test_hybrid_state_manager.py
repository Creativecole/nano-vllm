import pytest
import torch

from nanovllm.engine.layer_state import (
    DeltaNetStateSpec,
    HybridStateManager,
    delta_state_bytes_per_sequence,
)


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
    }
