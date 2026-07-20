from types import SimpleNamespace

import pytest


def tiny_qwen35_kwargs(layer_types=None, tie_word_embeddings=False):
    layer_types = layer_types or ["linear_attention", "full_attention"]
    return {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": len(layer_types),
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "hidden_act": "silu",
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "layer_types": list(layer_types),
        "pad_token_id": 0,
        "tie_word_embeddings": tie_word_embeddings,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [1, 1, 0],
        },
    }


@pytest.fixture
def tiny_config():
    return SimpleNamespace(**tiny_qwen35_kwargs())


@pytest.fixture
def hf_tiny_config():
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(**tiny_qwen35_kwargs())
    config._attn_implementation = "eager"
    return config


def error_metrics(actual, expected):
    difference = (actual.float() - expected.float()).abs()
    return difference.max().item(), difference.mean().item()


def use_transformers_recurrent_reference(modeling, linear_attention):
    def recurrent_chunk(
        query,
        key,
        value,
        g,
        beta,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=False,
        **_kwargs,
    ):
        return modeling.torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    linear_attention.chunk_gated_delta_rule = recurrent_chunk
