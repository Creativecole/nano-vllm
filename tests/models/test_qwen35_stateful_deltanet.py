import pytest
import torch

from nanovllm.engine.layer_state import DeltaNetState
from nanovllm.models.qwen3_5 import Qwen3_5GatedDeltaNet
from nanovllm.utils.context import reset_context, set_context
from conftest import error_metrics


def make_zero_state(layer, batch_size, dtype):
    return DeltaNetState(
        layer_idx=layer.layer_idx,
        conv_state=torch.zeros(
            batch_size,
            layer.conv_dim,
            layer.conv_kernel_size,
            dtype=dtype,
        ),
        recurrent_state=torch.zeros(
            batch_size,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_stateful_chunks_match_no_cache_forward(hf_tiny_config, batch_size):
    torch.manual_seed(31)
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    hidden_states = torch.randn(batch_size, 7, hf_tiny_config.hidden_size)
    state = make_zero_state(layer, batch_size, hidden_states.dtype)

    with torch.no_grad():
        expected, expected_recurrent = layer(hidden_states, return_state=True)
        first = layer._forward_stateful_chunk(hidden_states[:, :3], state)
        second = layer._forward_stateful_chunk(hidden_states[:, 3:6], state)
        third = layer._forward_stateful_chunk(hidden_states[:, 6:], state)
        output = torch.cat((first, second, third), dim=1)

    max_abs_error, mean_abs_error = error_metrics(output, expected)
    print(
        f"Qwen3.5 stateful DeltaNet max_abs_error={max_abs_error:.8f}, "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        state.recurrent_state, expected_recurrent, rtol=1e-5, atol=1e-6
    )


def test_stateful_batch_rows_do_not_share_recurrent_state(hf_tiny_config):
    torch.manual_seed(37)
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    hidden_states = torch.randn(2, 5, hf_tiny_config.hidden_size)
    state = make_zero_state(layer, 2, hidden_states.dtype)
    with torch.no_grad():
        layer._forward_stateful_chunk(hidden_states, state)
    assert not torch.equal(state.conv_state[0], state.conv_state[1])
    assert not torch.equal(state.recurrent_state[0], state.recurrent_state[1])


@pytest.mark.parametrize("batch_size,seq_len", [(1, 7), (2, 17)])
def test_chunked_stateful_prefill_matches_sequential(
    hf_tiny_config, batch_size, seq_len
):
    torch.manual_seed(43)
    hf_tiny_config.nanovllm_deltanet_backend = "sequential"
    sequential = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 8
    chunked = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    chunked.load_state_dict(sequential.state_dict(), strict=True)
    hidden_states = torch.randn(batch_size, seq_len, hf_tiny_config.hidden_size)
    sequential_state = make_zero_state(sequential, batch_size, hidden_states.dtype)
    chunked_state = make_zero_state(chunked, batch_size, hidden_states.dtype)

    with torch.no_grad():
        expected = sequential._forward_stateful_chunk(
            hidden_states, sequential_state
        )
        actual = chunked._forward_stateful_chunk(hidden_states, chunked_state)

    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(
        chunked_state.conv_state,
        sequential_state.conv_state,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        chunked_state.recurrent_state,
        sequential_state.recurrent_state,
        rtol=3e-4,
        atol=3e-5,
    )


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [1, 31, 64, 65, 127, 128, 129, 512])
def test_batched_chunked_stateful_prefill_matches_per_sequence_reference(
    hf_tiny_config, batch_size, seq_len
):
    torch.manual_seed(61 + batch_size + seq_len)
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 64
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    hidden_states = torch.randn(batch_size, seq_len, hf_tiny_config.hidden_size)
    initial_state = make_zero_state(layer, batch_size, hidden_states.dtype)
    batched_state = DeltaNetState(
        layer_idx=0,
        conv_state=initial_state.conv_state.clone(),
        recurrent_state=initial_state.recurrent_state.clone(),
    )
    reference_state = DeltaNetState(
        layer_idx=0,
        conv_state=initial_state.conv_state.clone(),
        recurrent_state=initial_state.recurrent_state.clone(),
    )

    with torch.no_grad():
        actual = layer._forward_stateful_chunk(hidden_states, batched_state)
        expected_rows = []
        for row in range(batch_size):
            row_state = DeltaNetState(
                layer_idx=0,
                conv_state=reference_state.conv_state[row : row + 1],
                recurrent_state=reference_state.recurrent_state[row : row + 1],
            )
            expected_rows.append(
                layer._forward_stateful_chunk(
                    hidden_states[row : row + 1], row_state
                )
            )
        expected = torch.cat(expected_rows, dim=0)

    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(
        batched_state.conv_state, reference_state.conv_state, rtol=0, atol=0
    )
    torch.testing.assert_close(
        batched_state.recurrent_state,
        reference_state.recurrent_state,
        rtol=3e-4,
        atol=3e-5,
    )


def test_equal_length_packed_prefill_uses_one_batched_recurrence(hf_tiny_config):
    torch.manual_seed(67)
    batch_size, seq_len = 4, 65
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 64
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    hidden_states = torch.randn(batch_size, seq_len, hf_tiny_config.hidden_size)
    packed = hidden_states.reshape(batch_size * seq_len, -1)
    state = make_zero_state(layer, batch_size, hidden_states.dtype)
    reference_state = make_zero_state(layer, batch_size, hidden_states.dtype)
    layer.set_diagnostics(True)

    try:
        set_context(True, prefill_seq_lens=(seq_len,) * batch_size)
        with torch.no_grad():
            actual = layer._forward_packed(packed, state)
            stats = layer.get_diagnostics()
            layer.set_diagnostics(False, reset=False)
            expected = layer._forward_stateful_chunk(hidden_states, reference_state)
    finally:
        reset_context()

    prefill_calls = [call for call in stats["calls"] if call["is_prefill"]]
    assert stats["equal_length_batched_prefill_calls"] == 1
    assert stats["variable_length_fallback_calls"] == 0
    assert len(prefill_calls) == 1
    assert prefill_calls[0]["query_shape"][0] == batch_size
    torch.testing.assert_close(
        actual, expected.reshape_as(actual), rtol=3e-4, atol=3e-5
    )
    torch.testing.assert_close(
        state.recurrent_state,
        reference_state.recurrent_state,
        rtol=3e-4,
        atol=3e-5,
    )


def test_variable_length_packed_prefill_keeps_correct_fallback(hf_tiny_config):
    torch.manual_seed(71)
    seq_lens = (31, 65, 129)
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 64
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    rows = [torch.randn(length, hf_tiny_config.hidden_size) for length in seq_lens]
    packed = torch.cat(rows, dim=0)
    state = make_zero_state(layer, len(rows), packed.dtype)
    reference_state = make_zero_state(layer, len(rows), packed.dtype)
    layer.set_diagnostics(True)

    try:
        set_context(True, prefill_seq_lens=seq_lens)
        with torch.no_grad():
            actual = layer._forward_packed(packed, state)
            stats = layer.get_diagnostics()
            layer.set_diagnostics(False, reset=False)
            expected_rows = []
            for row, hidden_states in enumerate(rows):
                row_state = DeltaNetState(
                    layer_idx=0,
                    conv_state=reference_state.conv_state[row : row + 1],
                    recurrent_state=reference_state.recurrent_state[row : row + 1],
                )
                expected_rows.append(
                    layer._forward_stateful_chunk(
                        hidden_states.unsqueeze(0), row_state
                    ).squeeze(0)
                )
            expected = torch.cat(expected_rows, dim=0)
    finally:
        reset_context()

    prefill_calls = [call for call in stats["calls"] if call["is_prefill"]]
    assert stats["equal_length_batched_prefill_calls"] == 0
    assert stats["variable_length_fallback_calls"] == 1
    assert stats["fallback_sequences"] == len(rows)
    assert len(prefill_calls) == len(rows)
    assert all(call["query_shape"][0] == 1 for call in prefill_calls)
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-5)
    torch.testing.assert_close(
        state.recurrent_state,
        reference_state.recurrent_state,
        rtol=3e-4,
        atol=3e-5,
    )


def test_batched_prefill_then_decode_matches_sequential_execution(hf_tiny_config):
    torch.manual_seed(73)
    batch_size, prompt_len, decode_steps = 4, 65, 3
    hf_tiny_config.nanovllm_deltanet_backend = "sequential"
    sequential = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 64
    chunked = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    chunked.load_state_dict(sequential.state_dict(), strict=True)
    prompt = torch.randn(batch_size, prompt_len, hf_tiny_config.hidden_size)
    decode_inputs = torch.randn(
        decode_steps, batch_size, 1, hf_tiny_config.hidden_size
    )
    sequential_state = make_zero_state(sequential, batch_size, prompt.dtype)
    chunked_state = make_zero_state(chunked, batch_size, prompt.dtype)

    with torch.no_grad():
        expected = [sequential._forward_stateful_chunk(prompt, sequential_state)]
        actual = [chunked._forward_stateful_chunk(prompt, chunked_state)]
        for step in range(decode_steps):
            expected.append(
                sequential._forward_stateful_chunk(
                    decode_inputs[step], sequential_state
                )
            )
            actual.append(
                chunked._forward_stateful_chunk(decode_inputs[step], chunked_state)
            )

    torch.testing.assert_close(
        torch.cat(actual, dim=1),
        torch.cat(expected, dim=1),
        rtol=3e-4,
        atol=3e-5,
    )
    torch.testing.assert_close(
        chunked_state.recurrent_state,
        sequential_state.recurrent_state,
        rtol=3e-4,
        atol=3e-5,
    )


def test_batched_chunked_prefill_preserves_bfloat16_io_and_fp32_state(
    hf_tiny_config,
):
    torch.manual_seed(79)
    batch_size, seq_len = 4, 65
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 64
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).to(torch.bfloat16)
    hidden_states = torch.randn(
        batch_size,
        seq_len,
        hf_tiny_config.hidden_size,
        dtype=torch.bfloat16,
    )
    batched_state = make_zero_state(layer, batch_size, hidden_states.dtype)
    reference_state = make_zero_state(layer, batch_size, hidden_states.dtype)

    with torch.no_grad():
        actual = layer._forward_stateful_chunk(hidden_states, batched_state)
        expected = torch.cat(
            [
                layer._forward_stateful_chunk(
                    hidden_states[row : row + 1],
                    DeltaNetState(
                        layer_idx=0,
                        conv_state=reference_state.conv_state[row : row + 1],
                        recurrent_state=reference_state.recurrent_state[
                            row : row + 1
                        ],
                    ),
                )
                for row in range(batch_size)
            ],
            dim=0,
        )

    assert actual.dtype == torch.bfloat16
    assert batched_state.recurrent_state.dtype == torch.float32
    assert torch.isfinite(actual).all()
    assert torch.isfinite(batched_state.recurrent_state).all()
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        batched_state.recurrent_state,
        reference_state.recurrent_state,
        rtol=3e-2,
        atol=3e-2,
    )
