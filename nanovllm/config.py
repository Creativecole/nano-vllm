import os
from dataclasses import dataclass
import torch
from transformers import AutoConfig


_DTYPE_ALIASES = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "torch.float16": torch.float16,
}


def resolve_torch_dtype(value: torch.dtype | str) -> torch.dtype:
    """Normalize supported Hugging Face dtype representations."""
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError(f"Unsupported dtype value {value!r} ({type(value).__name__})")
    dtype = _DTYPE_ALIASES.get(value.strip().lower())
    if dtype is None:
        supported = ", ".join(sorted(_DTYPE_ALIASES))
        raise ValueError(f"Unsupported dtype {value!r}; expected one of: {supported}")
    return dtype


def get_hf_text_config(hf_config):
    """Return the decoder text config for plain and wrapped HF configs."""
    text_config = getattr(hf_config, "text_config", None)
    if text_config is not None:
        return text_config
    get_text_config = getattr(hf_config, "get_text_config", None)
    if get_text_config is not None:
        try:
            return get_text_config(decoder=True)
        except TypeError:
            return get_text_config()
    return hf_config


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hybrid_state_capacity: int = 0
    hybrid_state_memory_fraction: float = 0.1
    enable_prefix_cache: bool = True
    hf_config: object | None = None
    hf_text_config: object | None = None
    dtype: torch.dtype | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.hybrid_state_capacity >= 0
        assert 0 < self.hybrid_state_memory_fraction < 1
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.hf_text_config = get_hf_text_config(self.hf_config)
        dtype_value = self.dtype
        if dtype_value is None:
            dtype_value = getattr(self.hf_text_config, "dtype", None)
        if dtype_value is None:
            dtype_value = getattr(self.hf_text_config, "torch_dtype", None)
        if dtype_value is None:
            raise ValueError("Model text config does not declare dtype or torch_dtype")
        self.dtype = resolve_torch_dtype(dtype_value)
        self.max_model_len = min(self.max_model_len, self.hf_text_config.max_position_embeddings)
