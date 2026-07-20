from types import SimpleNamespace

import pytest
import torch

from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
from nanovllm.utils.loader import WeightLoadingError, load_weights
from conftest import tiny_qwen35_kwargs


def _checkpoint_name(parameter_name):
    if parameter_name.startswith("model."):
        return "model.language_model." + parameter_name[len("model.") :]
    return parameter_name


def _checkpoint_weights(model):
    weights = []
    seen = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        weights.append((_checkpoint_name(name), torch.randn_like(parameter)))
    return weights


def test_strict_loader_maps_text_prefix_and_reports_non_text_skips():
    model = Qwen3_5ForCausalLM(SimpleNamespace(**tiny_qwen35_kwargs()))
    weights = _checkpoint_weights(model)
    weights += [
        ("model.visual.blocks.0.weight", torch.ones(1)),
        ("mtp.layers.0.weight", torch.ones(1)),
    ]
    report = load_weights(model, weights)
    assert report.ok
    assert not report.missing
    assert report.intentionally_skipped_non_text == {
        "model.visual.blocks.0.weight",
        "mtp.layers.0.weight",
    }


def test_strict_loader_rejects_missing_text_weight():
    model = Qwen3_5ForCausalLM(SimpleNamespace(**tiny_qwen35_kwargs()))
    weights = _checkpoint_weights(model)[1:]
    with pytest.raises(WeightLoadingError) as error:
        load_weights(model, weights)
    assert error.value.report.missing


def test_strict_loader_rejects_duplicate_and_unexpected_text_weight():
    model = Qwen3_5ForCausalLM(SimpleNamespace(**tiny_qwen35_kwargs()))
    weights = _checkpoint_weights(model)
    weights.append(weights[0])
    weights.append(("model.language_model.layers.0.not_a_weight", torch.ones(1)))
    with pytest.raises(WeightLoadingError) as error:
        load_weights(model, weights)
    assert error.value.report.duplicate
    assert error.value.report.unexpected_text_weights


def test_tied_embedding_is_loaded_once_without_false_duplicate():
    config = SimpleNamespace(**tiny_qwen35_kwargs(tie_word_embeddings=True))
    model = Qwen3_5ForCausalLM(config)
    weights = _checkpoint_weights(model)
    embedding_weight = next(
        weight for name, weight in weights if name == "model.language_model.embed_tokens.weight"
    )
    weights.append(("lm_head.weight", embedding_weight.clone()))
    report = load_weights(model, weights)
    assert report.ok
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not report.duplicate
    assert report.tied_aliases == {"lm_head.weight"}
