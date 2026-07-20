from types import SimpleNamespace

import pytest
import torch

from nanovllm.config import Config, get_hf_text_config, resolve_torch_dtype


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        ("bfloat16", torch.bfloat16),
        ("float16", torch.float16),
        ("torch.bfloat16", torch.bfloat16),
        ("torch.float16", torch.float16),
    ],
)
def test_resolve_torch_dtype(value, expected):
    assert resolve_torch_dtype(value) is expected


def test_resolve_torch_dtype_rejects_unsupported_values():
    with pytest.raises((TypeError, ValueError)):
        resolve_torch_dtype("float32")


def test_get_hf_text_config_prefers_wrapped_text_config():
    text = SimpleNamespace(hidden_size=4096)
    outer = SimpleNamespace(text_config=text, hidden_size=1)
    assert get_hf_text_config(outer) is text


def test_get_hf_text_config_keeps_plain_text_config():
    text = SimpleNamespace(hidden_size=4096)
    assert get_hf_text_config(text) is text


def test_config_preserves_outer_and_text_configs(monkeypatch, tmp_path):
    text = SimpleNamespace(
        model_type="qwen3_5_text",
        dtype="torch.bfloat16",
        max_position_embeddings=262144,
    )
    outer = SimpleNamespace(model_type="qwen3_5", text_config=text)
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda _model: outer,
    )

    config = Config(model=str(tmp_path), max_model_len=4096)
    assert config.hf_config is outer
    assert config.hf_text_config is text
    assert config.dtype is torch.bfloat16
    assert config.max_model_len == 4096


def test_explicit_config_dtype_accepts_string(monkeypatch, tmp_path):
    text = SimpleNamespace(dtype=torch.bfloat16, max_position_embeddings=128)
    monkeypatch.setattr(
        "nanovllm.config.AutoConfig.from_pretrained",
        lambda _model: text,
    )
    config = Config(model=str(tmp_path), dtype="torch.float16")
    assert config.dtype is torch.float16
