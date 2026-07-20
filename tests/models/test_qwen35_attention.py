import pytest
import torch

from nanovllm.models.qwen3_5 import Qwen3_5Attention, Qwen3_5RotaryEmbedding
from conftest import error_metrics


def test_rotary_embedding_matches_transformers(hf_tiny_config):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    reference = modeling.Qwen3_5TextRotaryEmbedding(hf_tiny_config)
    actual = Qwen3_5RotaryEmbedding(hf_tiny_config)
    hidden_states = torch.randn(2, 5, hf_tiny_config.hidden_size)
    positions = torch.arange(5).view(1, -1).expand(2, -1)
    with torch.no_grad():
        expected_cos, expected_sin = reference(hidden_states, positions)
        cos, sin = actual(hidden_states, positions)
    torch.testing.assert_close(cos, expected_cos, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(sin, expected_sin, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("batch_size,seq_len", [(1, 3), (2, 5)])
def test_full_attention_matches_transformers_reference(
    hf_tiny_config, batch_size, seq_len
):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.manual_seed(11)

    reference = modeling.Qwen3_5Attention(hf_tiny_config, layer_idx=1).eval()
    actual = Qwen3_5Attention(hf_tiny_config, layer_idx=1).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)

    hidden_states = torch.randn(batch_size, seq_len, hf_tiny_config.hidden_size)
    positions = torch.arange(seq_len).view(1, -1).expand(batch_size, -1)
    position_embeddings = Qwen3_5RotaryEmbedding(hf_tiny_config)(hidden_states, positions)
    causal_mask = torch.zeros(batch_size, 1, seq_len, seq_len)
    causal_mask.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1),
        float("-inf"),
    )

    with torch.no_grad():
        expected, _ = reference(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )
        output = actual(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
        )

    max_abs_error, mean_abs_error = error_metrics(output, expected)
    print(
        f"Qwen3.5 attention max_abs_error={max_abs_error:.8f}, "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)


def test_partial_rope_preserves_unrotated_dimensions(hf_tiny_config):
    from nanovllm.models.qwen3_5 import apply_partial_rotary_pos_emb

    batch_size, seq_len = 2, 5
    query = torch.randn(batch_size, 4, seq_len, hf_tiny_config.head_dim)
    key = torch.randn(batch_size, 2, seq_len, hf_tiny_config.head_dim)
    hidden = torch.zeros(batch_size, seq_len, hf_tiny_config.hidden_size)
    positions = torch.arange(seq_len).view(1, -1).expand(batch_size, -1)
    cos, sin = Qwen3_5RotaryEmbedding(hf_tiny_config)(hidden, positions)
    q_out, k_out = apply_partial_rotary_pos_emb(query, key, cos, sin)
    rotary_dim = cos.shape[-1]
    torch.testing.assert_close(q_out[..., rotary_dim:], query[..., rotary_dim:])
    torch.testing.assert_close(k_out[..., rotary_dim:], key[..., rotary_dim:])
