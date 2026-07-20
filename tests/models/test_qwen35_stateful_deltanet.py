import pytest
import torch

from nanovllm.engine.layer_state import DeltaNetState
from nanovllm.models.qwen3_5 import Qwen3_5GatedDeltaNet
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
