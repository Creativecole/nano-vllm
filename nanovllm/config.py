import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


_DTYPE_MAP = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


_MODEL_ATTRS = (
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "hidden_act",
    "rms_norm_eps",
    "attention_bias",
    "head_dim",
    "rope_theta",
    "rope_scaling",
    "tie_word_embeddings",
    "max_position_embeddings",
    "max_sequence_length",
    "max_seq_len",
    "seq_length",
    "n_positions",
    "model_max_length",
)


def iter_config_candidates(hf_config):
    yield hf_config
    for attr in ("text_config", "llm_config", "language_config", "decoder_config", "model_config"):
        value = getattr(hf_config, attr, None)
        if value is not None and value is not hf_config:
            yield value


def get_first_config_attr(hf_config, attr: str, default=None):
    for config in iter_config_candidates(hf_config):
        value = getattr(config, attr, None)
        if value is not None:
            return value
    return default


def normalize_text_config_attrs(hf_config):
    for attr in _MODEL_ATTRS:
        if getattr(hf_config, attr, None) is None:
            value = get_first_config_attr(hf_config, attr)
            if value is not None:
                setattr(hf_config, attr, value)

    if getattr(hf_config, "tie_word_embeddings", None) is None:
        hf_config.tie_word_embeddings = False

    missing = [attr for attr in _MODEL_ATTRS[:8] if getattr(hf_config, attr, None) is None]
    if missing:
        raise AttributeError(
            f"Model config is missing required text-model attributes: {missing}. "
            "If this is a non-dense or wrapped architecture, add an explicit model adapter."
        )


def infer_max_position_embeddings(hf_config, default: int) -> int:
    for attr in (
        "max_position_embeddings",
        "max_sequence_length",
        "max_seq_len",
        "seq_length",
        "n_positions",
        "model_max_length",
    ):
        value = getattr(hf_config, attr, None)
        if isinstance(value, int) and value > 0:
            return value

    rope_scaling = get_first_config_attr(hf_config, "rope_scaling")
    if isinstance(rope_scaling, dict):
        for key in ("max_position_embeddings", "original_max_position_embeddings"):
            value = rope_scaling.get(key)
            if isinstance(value, int) and value > 0:
                return value

    return default


def coerce_torch_dtype(value) -> torch.dtype | None:
    if isinstance(value, torch.dtype) and value.is_floating_point:
        return value
    if isinstance(value, str):
        key = value.removeprefix("torch.").lower()
        return _DTYPE_MAP.get(key)
    return None


def infer_torch_dtype(hf_config, default: torch.dtype = torch.bfloat16) -> torch.dtype:
    dtype = coerce_torch_dtype(get_first_config_attr(hf_config, "dtype"))
    if dtype is not None:
        return dtype
    for config in iter_config_candidates(hf_config):
        # Newer Transformers emits a deprecation warning when the torch_dtype
        # property is accessed. Read the raw config dict for older checkpoints
        # that still serialize this field.
        dtype = coerce_torch_dtype(getattr(config, "__dict__", {}).get("torch_dtype"))
        if dtype is not None:
            return dtype
    return default


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    norm_backend: str = "triton"
    activation_backend: str = "triton"
    rope_backend: str = "triton"
    linear_backend: str = "torch"
    attn_backend: str = "flash_attn"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.norm_backend in {"auto", "torch", "triton"}
        assert self.activation_backend in {"auto", "torch", "triton"}
        assert self.rope_backend in {"auto", "torch", "triton"}
        assert self.linear_backend in {"auto", "torch", "triton", "cuda"}
        assert self.attn_backend in {
            "torch_sdpa",
            "flash_attn",
            "torch_paged",
            "triton_paged_decode",
            "triton_paged_decode_v2",
        }
        if self.attn_backend == "torch_sdpa":
            raise ValueError("attn_backend='torch_sdpa' is a prefill benchmark backend, not an e2e runtime backend")
        if self.attn_backend in {"torch_paged", "triton_paged_decode", "triton_paged_decode_v2"} and not self.enforce_eager:
            raise ValueError(
                f"attn_backend='{self.attn_backend}' currently requires enforce_eager=True; "
                "CUDA Graph integration is intentionally left for a later step"
            )
        self.hf_config = AutoConfig.from_pretrained(self.model)
        normalize_text_config_attrs(self.hf_config)
        max_position_embeddings = infer_max_position_embeddings(self.hf_config, self.max_model_len)
        self.hf_config.max_position_embeddings = max_position_embeddings
        dtype = infer_torch_dtype(self.hf_config)
        self.hf_config.dtype = dtype
        self.hf_config.attn_backend = self.attn_backend
        self.hf_config.kvcache_block_size = self.kvcache_block_size
        self.max_model_len = min(self.max_model_len, max_position_embeddings)
