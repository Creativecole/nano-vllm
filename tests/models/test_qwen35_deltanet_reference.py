import pytest
import torch

from nanovllm.models.qwen3_5 import (
    Qwen3_5GatedDeltaNet,
    gated_delta_rule_reference,
)
from conftest import error_metrics, use_transformers_recurrent_reference


def test_sequential_delta_rule_state_shape_and_finiteness():
    torch.manual_seed(3)
    batch_size, seq_len, heads, key_dim, value_dim = 2, 7, 4, 8, 8
    query = torch.randn(batch_size, seq_len, heads, key_dim, dtype=torch.bfloat16)
    key = torch.randn_like(query)
    value = torch.randn(batch_size, seq_len, heads, value_dim, dtype=torch.bfloat16)
    decay = -torch.rand(batch_size, seq_len, heads)
    beta = torch.sigmoid(torch.randn(batch_size, seq_len, heads))

    output, state = gated_delta_rule_reference(query, key, value, decay, beta)
    assert output.shape == value.shape
    assert state.shape == (batch_size, heads, key_dim, value_dim)
    assert state.dtype == torch.float32
    assert torch.isfinite(output).all()
    assert torch.isfinite(state).all()


@pytest.mark.parametrize("batch_size,seq_len", [(1, 3), (2, 7)])
def test_deltanet_layer_matches_transformers_recurrent_reference(
    hf_tiny_config, batch_size, seq_len
):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.manual_seed(17)

    reference = modeling.Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    # Force the official clear PyTorch recurrence even if optional fused kernels
    # happen to be installed on the test machine.
    use_transformers_recurrent_reference(modeling, reference)
    actual = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    hidden_states = torch.randn(batch_size, seq_len, hf_tiny_config.hidden_size)
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    with torch.no_grad():
        expected = reference(hidden_states, attention_mask=attention_mask)
        output = actual(hidden_states, attention_mask=attention_mask)

    max_abs_error, mean_abs_error = error_metrics(output, expected)
    print(
        f"Qwen3.5 DeltaNet max_abs_error={max_abs_error:.8f}, "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_deltanet_supports_bfloat16_without_nan(hf_tiny_config):
    layer = Qwen3_5GatedDeltaNet(hf_tiny_config, layer_idx=0).to(torch.bfloat16)
    hidden_states = torch.randn(2, 5, hf_tiny_config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        output = layer(hidden_states)
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output).all()
