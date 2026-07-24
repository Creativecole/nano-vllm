from nanovllm.attention.backend.delta_attention import DeltaNetBackend
from nanovllm.attention.backend.full_attention import FullAttentionBackend
from nanovllm.attention.backend.registry import (
    create_attention_backend,
    get_attention_backend_registration,
    register_attention_backend,
)

register_attention_backend(
    layer_type="full_attention",
    name="flash_attn",
    backend_cls=FullAttentionBackend,
    module_attr="self_attn",
    default=True,
)
register_attention_backend(
    layer_type="linear_attention",
    name="deltanet_reference",
    backend_cls=DeltaNetBackend,
    module_attr="linear_attn",
    default=True,
)

__all__ = [
    "DeltaNetBackend",
    "FullAttentionBackend",
    "create_attention_backend",
    "get_attention_backend_registration",
    "register_attention_backend",
]
