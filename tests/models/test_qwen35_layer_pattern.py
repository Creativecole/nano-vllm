from types import SimpleNamespace

import pytest

from nanovllm.engine.layer_state import DeltaNetStateSpec, PagedKVStateSpec
from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5Model
from conftest import tiny_qwen35_kwargs


def test_layer_pattern_is_driven_by_config():
    pattern = ["full_attention", "linear_attention", "linear_attention", "full_attention"]
    model = Qwen3_5Model(SimpleNamespace(**tiny_qwen35_kwargs(pattern)))
    assert [layer.block_type for layer in model.layers] == pattern
    assert hasattr(model.layers[0], "self_attn")
    assert hasattr(model.layers[1], "linear_attn")


def test_layer_pattern_length_must_match_layer_count():
    config = SimpleNamespace(**tiny_qwen35_kwargs(["full_attention"]))
    config.num_hidden_layers = 2
    with pytest.raises(ValueError, match="layer_types length"):
        Qwen3_5Model(config)


def test_unknown_layer_type_is_rejected():
    config = SimpleNamespace(**tiny_qwen35_kwargs(["mystery_attention"]))
    with pytest.raises(ValueError, match="Unsupported Qwen3.5 layer type"):
        Qwen3_5Model(config)


def test_layer_state_specs_follow_config_pattern():
    pattern = ["linear_attention", "full_attention", "linear_attention"]
    config = SimpleNamespace(**tiny_qwen35_kwargs(pattern))
    model = Qwen3_5ForCausalLM(config)
    specs = model.get_layer_state_specs()
    assert [spec.layer_type for spec in specs] == pattern
    assert isinstance(specs[0], DeltaNetStateSpec)
    assert isinstance(specs[1], PagedKVStateSpec)
    assert specs[0].conv_width == config.linear_conv_kernel_dim
    assert specs[0].key_head_dim == config.linear_key_head_dim
    assert specs[1].num_kv_heads == config.num_key_value_heads
    assert specs[1].head_dim == config.head_dim
