import os
from dataclasses import dataclass
from transformers import AutoConfig


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

    rope_scaling = getattr(hf_config, "rope_scaling", None)
    if isinstance(rope_scaling, dict):
        for key in ("max_position_embeddings", "original_max_position_embeddings"):
            value = rope_scaling.get(key)
            if isinstance(value, int) and value > 0:
                return value

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
    kv_cache_dtype: str = "auto"
    kv_cache_scale: float = 1.0
    resolved_kv_cache_dtype: str = "bf16"
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.kv_cache_dtype in {"auto", "bf16", "fp8_e4m3"}
        assert self.kv_cache_scale > 0
        self.hf_config = AutoConfig.from_pretrained(self.model)
        max_position_embeddings = infer_max_position_embeddings(self.hf_config, self.max_model_len)
        self.hf_config.max_position_embeddings = max_position_embeddings
        self.max_model_len = min(self.max_model_len, max_position_embeddings)
