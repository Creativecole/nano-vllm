import pytest
import torch

from nanovllm.engine.layer_state import DeltaNetStateSpec, HybridStateManager
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.utils.context import reset_context, set_context
from conftest import (
    TorchPagedAttentionReference,
    error_metrics,
    use_transformers_recurrent_reference,
)


def make_block_metadata(lengths, block_size):
    max_blocks = max((length + block_size - 1) // block_size for length in lengths)
    block_tables = torch.full((len(lengths), max_blocks), -1, dtype=torch.int32)
    slots = []
    next_block = 0
    for row, length in enumerate(lengths):
        num_blocks = (length + block_size - 1) // block_size
        blocks = torch.arange(next_block, next_block + num_blocks, dtype=torch.int32)
        block_tables[row, :num_blocks] = blocks
        for position in range(length):
            slots.append(int(blocks[position // block_size]) * block_size + position % block_size)
        next_block += num_blocks
    return block_tables, slots, next_block


def install_torch_paged_attention(model, num_blocks, block_size):
    for layer in model.model.layers:
        if layer.block_type != "full_attention":
            continue
        attention = layer.self_attn
        attention.paged_attention = TorchPagedAttentionReference(
            num_blocks,
            block_size,
            attention.num_heads,
            attention.num_kv_heads,
            attention.head_dim,
            attention.scaling,
        )


def official_last_logits(reference, prompts):
    rows = []
    with torch.no_grad():
        for prompt in prompts:
            input_ids = torch.tensor(prompt).view(1, -1)
            rows.append(reference(input_ids=input_ids, use_cache=False).logits[:, -1])
    return torch.cat(rows, dim=0)


def make_hybrid_pair(hf_tiny_config, batch_size, lengths):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    reference = modeling.Qwen3_5ForCausalLM(hf_tiny_config).eval()
    for layer in reference.model.layers:
        if hasattr(layer, "linear_attn"):
            use_transformers_recurrent_reference(modeling, layer.linear_attn)
    model = Qwen3_5ForCausalLM(hf_tiny_config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)
    block_size = 4
    block_tables, _, num_blocks = make_block_metadata(lengths, block_size)
    install_torch_paged_attention(model, num_blocks, block_size)
    specs = [
        spec for spec in model.get_layer_state_specs() if isinstance(spec, DeltaNetStateSpec)
    ]
    manager = HybridStateManager(specs, capacity=batch_size, device="cpu")
    return model, reference, manager, block_tables, block_size


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_hybrid_prefill_and_decode_match_transformers(hf_tiny_config, batch_size):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.manual_seed(41)
    reference = modeling.Qwen3_5ForCausalLM(hf_tiny_config).eval()
    for layer in reference.model.layers:
        if hasattr(layer, "linear_attn"):
            use_transformers_recurrent_reference(modeling, layer.linear_attn)
    model = Qwen3_5ForCausalLM(hf_tiny_config).eval()
    model.load_state_dict(reference.state_dict(), strict=True)

    prompts = [list(range(1 + row, 4 + row * 2)) for row in range(batch_size)]
    lengths = [len(prompt) for prompt in prompts]
    block_size = 4
    block_tables, _, num_blocks = make_block_metadata(
        [length + 2 for length in lengths], block_size
    )
    install_torch_paged_attention(model, num_blocks, block_size)
    delta_specs = [
        spec for spec in model.get_layer_state_specs() if isinstance(spec, DeltaNetStateSpec)
    ]
    manager = HybridStateManager(delta_specs, capacity=batch_size, device="cpu")
    seq_ids = list(range(batch_size))
    manager.allocate(seq_ids)

    packed_ids = torch.tensor([token for prompt in prompts for token in prompt])
    packed_positions = torch.tensor(
        [position for length in lengths for position in range(length)]
    )
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    prefill_slots = []
    for row, length in enumerate(lengths):
        for position in range(length):
            block = int(block_tables[row, position // block_size])
            prefill_slots.append(block * block_size + position % block_size)
    states = manager.gather(seq_ids)
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_k=max(lengths),
        slot_mapping=torch.tensor(prefill_slots, dtype=torch.int32),
        prefill_seq_lens=tuple(lengths),
    )
    with torch.no_grad():
        hidden = model(packed_ids, packed_positions, layer_states=states)
        logits = model.compute_logits(hidden)
    manager.commit(states)
    expected = official_last_logits(reference, prompts)
    max_abs_error, mean_abs_error = error_metrics(logits, expected)
    print(
        f"Qwen3.5 hybrid prefill max_abs_error={max_abs_error:.8f}, "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    torch.testing.assert_close(logits, expected, rtol=3e-5, atol=3e-6)
    first_tokens = logits.argmax(dim=-1)
    assert torch.equal(first_tokens, expected.argmax(dim=-1))

    decode_positions = torch.tensor(lengths)
    decode_slots = []
    for row, position in enumerate(lengths):
        block = int(block_tables[row, position // block_size])
        decode_slots.append(block * block_size + position % block_size)
    states = manager.gather(list(reversed(seq_ids)))
    reversed_tokens = first_tokens.flip(0)
    reversed_positions = decode_positions.flip(0)
    reversed_tables = block_tables.flip(0)
    reversed_lengths = [lengths[index] + 1 for index in reversed(range(batch_size))]
    reversed_slots = list(reversed(decode_slots))
    set_context(
        False,
        slot_mapping=torch.tensor(reversed_slots, dtype=torch.int32),
        context_lens=torch.tensor(reversed_lengths, dtype=torch.int32),
        block_tables=reversed_tables,
    )
    with torch.no_grad():
        hidden = model(reversed_tokens, reversed_positions, layer_states=states)
        decode_logits = model.compute_logits(hidden)
    expected_decode = official_last_logits(
        reference,
        [prompts[index] + [int(first_tokens[index])] for index in reversed(range(batch_size))],
    )
    max_abs_error, mean_abs_error = error_metrics(decode_logits, expected_decode)
    print(
        f"Qwen3.5 hybrid decode max_abs_error={max_abs_error:.8f}, "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    torch.testing.assert_close(decode_logits, expected_decode, rtol=3e-5, atol=3e-6)
    assert torch.equal(decode_logits.argmax(-1), expected_decode.argmax(-1))
    reset_context()


def test_compacted_reordered_decode_matches_independent_greedy(hf_tiny_config):
    torch.manual_seed(43)
    prompts = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    generation_lengths = [1, 3, 2]
    capacity_lengths = [
        len(prompt) + generation_length
        for prompt, generation_length in zip(prompts, generation_lengths)
    ]
    model, reference, manager, block_tables, block_size = make_hybrid_pair(
        hf_tiny_config, len(prompts), capacity_lengths
    )
    seq_ids = [100, 200, 300]
    manager.allocate(seq_ids)

    prompt_lengths = [len(prompt) for prompt in prompts]
    packed_ids = torch.tensor([token for prompt in prompts for token in prompt])
    packed_positions = torch.tensor(
        [position for length in prompt_lengths for position in range(length)]
    )
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(prompt_lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    prefill_slots = []
    for row, length in enumerate(prompt_lengths):
        for position in range(length):
            block = int(block_tables[row, position // block_size])
            prefill_slots.append(block * block_size + position % block_size)

    states = manager.gather(seq_ids)
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max(prompt_lengths),
        max_seqlen_k=max(prompt_lengths),
        slot_mapping=torch.tensor(prefill_slots, dtype=torch.int32),
        prefill_seq_lens=tuple(prompt_lengths),
    )
    with torch.no_grad():
        generated = model.compute_logits(
            model(packed_ids, packed_positions, layer_states=states)
        ).argmax(-1).tolist()
    manager.commit(states)
    generated_by_row = [[token] for token in generated]

    while any(
        len(tokens) < target
        for tokens, target in zip(generated_by_row, generation_lengths)
    ):
        active_rows = [
            row
            for row, target in enumerate(generation_lengths)
            if len(generated_by_row[row]) < target
        ]
        active_rows.reverse()
        active_seq_ids = [seq_ids[row] for row in active_rows]
        positions = [
            len(prompts[row]) + len(generated_by_row[row]) - 1 for row in active_rows
        ]
        slots = []
        for row, position in zip(active_rows, positions):
            block = int(block_tables[row, position // block_size])
            slots.append(block * block_size + position % block_size)
        states = manager.gather(active_seq_ids)
        set_context(
            False,
            slot_mapping=torch.tensor(slots, dtype=torch.int32),
            context_lens=torch.tensor([position + 1 for position in positions], dtype=torch.int32),
            block_tables=block_tables[active_rows],
        )
        input_ids = torch.tensor([generated_by_row[row][-1] for row in active_rows])
        with torch.no_grad():
            next_tokens = model.compute_logits(
                model(input_ids, torch.tensor(positions), layer_states=states)
            ).argmax(-1).tolist()
        manager.commit(states)
        for row, token in zip(active_rows, next_tokens):
            generated_by_row[row].append(token)

    expected_by_row = []
    for row, prompt in enumerate(prompts):
        tokens = []
        for _ in range(generation_lengths[row]):
            logits = official_last_logits(reference, [prompt + tokens])
            tokens.append(int(logits.argmax(-1)))
        expected_by_row.append(tokens)

    assert generated_by_row == expected_by_row
    reset_context()
