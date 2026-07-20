from types import SimpleNamespace

import pytest

from nanovllm.models.qwen3_5 import Qwen3_5Model
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
