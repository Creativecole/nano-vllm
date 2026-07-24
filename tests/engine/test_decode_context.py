import pytest
import torch

from nanovllm.attention import HybridAttentionMetadataBuilder
from nanovllm.engine.decode_context import HybridDecodeContext
from nanovllm.utils.context import Context, get_context, reset_context


class FakeSequence:
    def __init__(self, seq_id, length, block_table, temperature=0.0):
        self.seq_id = seq_id
        self.num_tokens = length
        self.block_table = list(block_table)
        self.temperature = temperature

    def __len__(self):
        return self.num_tokens


def make_context(requested, execution, mode="decode"):
    positions = torch.tensor([len(seq) - 1 for seq in execution])
    slot_mapping = torch.tensor([100, 200], dtype=torch.int32)
    context_lens = torch.tensor([len(seq) for seq in execution], dtype=torch.int32)
    block_tables = torch.tensor(
        [seq.block_table for seq in execution],
        dtype=torch.int32,
    )
    if mode == "decode":
        runtime_context = Context(
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
    else:
        runtime_context = Context(
            is_mixed=True,
            slot_mapping=slot_mapping,
            num_decode_requests=len(execution),
            decode_context_lens=context_lens,
            decode_block_tables=block_tables,
            sequence_query_lens=(1,) * len(execution),
        )
    metadata = HybridAttentionMetadataBuilder().build(
        context=runtime_context,
        request_ids=[seq.seq_id for seq in execution],
        query_lens=[1] * len(execution),
        is_prefilling=[False] * len(execution),
        positions=positions,
    )
    return HybridDecodeContext.create(
        mode=mode,
        requested_seqs=requested,
        execution_seqs=execution,
        state_layout_version=3,
        layer_states={1: object()},
        attention_metadata=metadata,
        temperatures=torch.zeros(len(execution)),
    )


def test_decode_context_reuses_stable_batch_and_advances_device_metadata():
    seq_a = FakeSequence(10, 5, [3])
    seq_b = FakeSequence(20, 9, [8])
    requested = [seq_b, seq_a]
    execution = [seq_a, seq_b]
    context = make_context(requested, execution)
    positions_storage = context.positions.data_ptr()
    lengths_storage = context.context_lens.data_ptr()
    slots_storage = context.slot_mapping.data_ptr()
    context.record_sampled_tokens(torch.tensor([31, 32]))

    seq_a.num_tokens += 1
    seq_b.num_tokens += 1
    assert context.reuse_failure(
        requested,
        mode="decode",
        state_layout_version=3,
    ) is None

    next_ids, positions = context.advance(execution)
    assert torch.equal(next_ids, torch.tensor([31, 32]))
    assert torch.equal(positions, torch.tensor([5, 9]))
    assert torch.equal(context.context_lens, torch.tensor([6, 10], dtype=torch.int32))
    assert torch.equal(context.slot_mapping, torch.tensor([101, 201], dtype=torch.int32))
    assert context.positions.data_ptr() == positions_storage
    assert context.context_lens.data_ptr() == lengths_storage
    assert context.slot_mapping.data_ptr() == slots_storage


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("batch", "batch_changed"),
        ("state", "state_layout_changed"),
        ("progress", "sequence_progress_changed"),
        ("block", "kv_block_layout_changed"),
        ("block_remap", "kv_block_layout_changed"),
        ("sampling", "sampling_changed"),
    ],
)
def test_decode_context_rejects_dynamic_batch_or_layout_changes(
    mutation,
    expected,
):
    seq_a = FakeSequence(10, 5, [3])
    seq_b = FakeSequence(20, 9, [8])
    requested = [seq_a, seq_b]
    context = make_context(requested, requested)
    context.record_sampled_tokens(torch.tensor([31, 32]))
    seq_a.num_tokens += 1
    seq_b.num_tokens += 1
    state_layout_version = 3

    if mutation == "batch":
        requested = [seq_b, seq_a]
    elif mutation == "state":
        state_layout_version += 1
    elif mutation == "progress":
        seq_b.num_tokens += 1
    elif mutation == "block":
        seq_b.block_table.append(9)
    elif mutation == "block_remap":
        seq_b.block_table[0] = 9
    elif mutation == "sampling":
        seq_b.temperature = 0.5

    assert context.reuse_failure(
        requested,
        mode="decode",
        state_layout_version=state_layout_version,
    ) == expected


def test_unified_decode_context_restores_mixed_runtime_metadata():
    seq_a = FakeSequence(10, 5, [3])
    seq_b = FakeSequence(20, 9, [8])
    context = make_context([seq_a, seq_b], [seq_a, seq_b], "unified_decode")

    try:
        context.activate_runtime_context()
        runtime_context = get_context()
        assert runtime_context.is_mixed
        assert runtime_context.num_decode_requests == 2
        assert runtime_context.sequence_query_lens == (1, 1)
        assert torch.equal(runtime_context.sample_indices, torch.tensor([0, 1]))
        assert runtime_context.decode_context_lens is context.context_lens
        assert runtime_context.decode_block_tables is context.block_tables
    finally:
        reset_context()


def test_decode_context_rejects_sampled_token_shape_mismatch():
    seq = FakeSequence(10, 5, [3])
    context = make_context(
        [seq, FakeSequence(20, 9, [8])],
        [seq, FakeSequence(20, 9, [8])],
    )

    with pytest.raises(ValueError, match="sampled-token shape"):
        context.record_sampled_tokens(torch.tensor([[31, 32]]))
