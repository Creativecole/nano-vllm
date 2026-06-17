from dataclasses import dataclass

from transformers import AutoConfig

from nanovllm.config import infer_max_position_embeddings, normalize_text_config_attrs


@dataclass(frozen=True)
class ModelShapes:
    model_name: str
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 151936
    max_position_embeddings: int = 4096
    qkv_dim: int = 4096
    o_proj_in: int = 2048

    @classmethod
    def default(cls):
        return cls(
            model_name="qwen3_0.6b_like",
            qkv_dim=1280,
            o_proj_in=1024,
        )

    @classmethod
    def from_model(cls, model_path: str):
        hf_config = AutoConfig.from_pretrained(model_path)
        normalize_text_config_attrs(hf_config)
        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None:
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads
        max_position_embeddings = infer_max_position_embeddings(hf_config, 4096)
        qkv_dim = (hf_config.num_attention_heads + 2 * hf_config.num_key_value_heads) * head_dim
        o_proj_in = hf_config.num_attention_heads * head_dim
        return cls(
            model_name=model_path,
            hidden_size=hf_config.hidden_size,
            intermediate_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            head_dim=head_dim,
            vocab_size=hf_config.vocab_size,
            max_position_embeddings=max_position_embeddings,
            qkv_dim=qkv_dim,
            o_proj_in=o_proj_in,
        )

    def describe(self) -> str:
        return (
            f"hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
            f"heads={self.num_attention_heads}, kv_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim}, vocab={self.vocab_size}, max_pos={self.max_position_embeddings}"
        )
