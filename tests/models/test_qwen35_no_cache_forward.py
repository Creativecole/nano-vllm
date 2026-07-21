import pytest
import torch

from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from conftest import error_metrics, use_transformers_recurrent_reference


def test_no_cache_logits_match_transformers(hf_tiny_config):
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    torch.manual_seed(23)

    reference = modeling.Qwen3_5ForCausalLM(hf_tiny_config).eval()
    for layer in reference.model.layers:
        if hasattr(layer, "linear_attn"):
            use_transformers_recurrent_reference(modeling, layer.linear_attn)

    actual = Qwen3_5ForCausalLM(hf_tiny_config).eval()
    actual.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.randint(1, hf_tiny_config.vocab_size, (2, 5))
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    with torch.no_grad():
        expected = reference(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        output = actual.forward_logits(input_ids, attention_mask=attention_mask)

    max_abs_error, mean_abs_error = error_metrics(output, expected)
    print(
        f"Qwen3.5 no-cache logits max_abs_error={max_abs_error:.8f}, "
        f"mean_abs_error={mean_abs_error:.8f}"
    )
    assert output.shape == (2, 5, hf_tiny_config.vocab_size)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, expected, rtol=2e-5, atol=2e-6)


def test_no_cache_forward_accepts_rank_one_input(hf_tiny_config):
    model = Qwen3_5ForCausalLM(hf_tiny_config).eval()
    input_ids = torch.randint(1, hf_tiny_config.vocab_size, (5,))
    with torch.no_grad():
        logits = model.forward_logits(input_ids)
    assert logits.shape == (5, hf_tiny_config.vocab_size)
    assert torch.isfinite(logits).all()


def test_chunked_no_cache_logits_match_sequential(hf_tiny_config):
    torch.manual_seed(47)
    hf_tiny_config.nanovllm_deltanet_backend = "sequential"
    sequential = Qwen3_5ForCausalLM(hf_tiny_config).eval()
    hf_tiny_config.nanovllm_deltanet_backend = "chunked"
    hf_tiny_config.nanovllm_deltanet_chunk_size = 4
    chunked = Qwen3_5ForCausalLM(hf_tiny_config).eval()
    chunked.load_state_dict(sequential.state_dict(), strict=True)
    input_ids = torch.randint(1, hf_tiny_config.vocab_size, (2, 9))

    with torch.no_grad():
        expected = sequential.forward_logits(input_ids)
        actual = chunked.forward_logits(input_ids)

    assert torch.equal(actual[:, -1].argmax(-1), expected[:, -1].argmax(-1))
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
