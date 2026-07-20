from types import SimpleNamespace

import pytest

from nanovllm.models.registry import resolve_model_registration


def test_registry_resolves_qwen3_by_architecture():
    outer = SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="unknown")
    registration = resolve_model_registration(outer)
    assert registration.class_name == "Qwen3ForCausalLM"


def test_registry_resolves_wrapped_qwen35_from_text_model_type():
    outer = SimpleNamespace(architectures=[], model_type="qwen3_5")
    text = SimpleNamespace(model_type="qwen3_5_text")
    registration = resolve_model_registration(outer, text)
    assert registration.class_name == "Qwen3_5ForCausalLM"


def test_registry_rejects_unknown_model():
    config = SimpleNamespace(architectures=["UnknownForCausalLM"], model_type="unknown")
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        resolve_model_registration(config)
