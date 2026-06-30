from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoConfig

from nanovllm.config import (
    infer_max_position_embeddings,
    infer_torch_dtype,
    normalize_text_config_attrs,
)


@dataclass(frozen=True)
class QwenAttentionShapes:
    model_name: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    dtype: str
    gqa_ratio: int
    q_proj_shape: tuple[int, int]
    k_proj_shape: tuple[int, int]
    v_proj_shape: tuple[int, int]
    o_proj_shape: tuple[int, int]
    qkv_proj_shape: tuple[int, int]
    gate_up_proj_shape: tuple[int, int]
    down_proj_shape: tuple[int, int]

    @property
    def q_heads(self) -> int:
        return self.num_attention_heads

    @property
    def kv_heads(self) -> int:
        return self.num_key_value_heads

    def decode_attention_shape(self, batch_size: int, seq_len: int, block_size: int) -> dict[str, Any]:
        max_blocks_per_seq = (seq_len + block_size - 1) // block_size
        return {
            "q": [batch_size, self.num_attention_heads, self.head_dim],
            "k_cache": ["num_blocks", block_size, self.num_key_value_heads, self.head_dim],
            "v_cache": ["num_blocks", block_size, self.num_key_value_heads, self.head_dim],
            "block_tables": [batch_size, max_blocks_per_seq],
            "context_lens": [batch_size],
            "block_size": block_size,
            "max_blocks_per_seq": max_blocks_per_seq,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_attr(config: Any, name: str, default: Any = None) -> Any:
    value = getattr(config, name, None)
    if value is not None:
        return value
    for nested in ("text_config", "llm_config", "language_config", "decoder_config", "model_config"):
        nested_config = getattr(config, nested, None)
        value = getattr(nested_config, name, None) if nested_config is not None else None
        if value is not None:
            return value
    return default


def _dtype_name(dtype: Any) -> str:
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    return str(dtype)


def extract_qwen_attention_shapes(config: Any, model_name: str = "config") -> QwenAttentionShapes:
    normalize_text_config_attrs(config)
    hidden_size = int(_get_attr(config, "hidden_size"))
    num_attention_heads = int(_get_attr(config, "num_attention_heads"))
    num_key_value_heads = int(_get_attr(config, "num_key_value_heads", num_attention_heads))
    head_dim = _get_attr(config, "head_dim")
    head_dim = int(head_dim) if head_dim is not None else hidden_size // num_attention_heads
    intermediate_size = int(_get_attr(config, "intermediate_size"))
    num_hidden_layers = int(_get_attr(config, "num_hidden_layers"))
    vocab_size = int(_get_attr(config, "vocab_size"))
    max_position_embeddings = infer_max_position_embeddings(config, 4096)
    dtype = infer_torch_dtype(config)
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError(
            f"num_attention_heads ({num_attention_heads}) must be divisible by "
            f"num_key_value_heads ({num_key_value_heads}) for GQA"
        )
    q_dim = num_attention_heads * head_dim
    kv_dim = num_key_value_heads * head_dim
    return QwenAttentionShapes(
        model_name=model_name,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        vocab_size=vocab_size,
        max_position_embeddings=max_position_embeddings,
        dtype=_dtype_name(dtype),
        gqa_ratio=num_attention_heads // num_key_value_heads,
        q_proj_shape=(q_dim, hidden_size),
        k_proj_shape=(kv_dim, hidden_size),
        v_proj_shape=(kv_dim, hidden_size),
        o_proj_shape=(hidden_size, q_dim),
        qkv_proj_shape=(q_dim + 2 * kv_dim, hidden_size),
        gate_up_proj_shape=(2 * intermediate_size, hidden_size),
        down_proj_shape=(hidden_size, intermediate_size),
    )


def load_qwen_attention_shapes(model: str) -> QwenAttentionShapes:
    config = AutoConfig.from_pretrained(model)
    return extract_qwen_attention_shapes(config, model_name=model)

