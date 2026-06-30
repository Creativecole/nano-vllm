"""KV-cache store wrapper.

The runtime implementation currently lives in `nanovllm.layers.attention` because the original
nano-vLLM layer calls it directly. This wrapper gives the attention-backend project a stable kernel
namespace without duplicating the existing implementation.
"""

from nanovllm.layers.attention import store_kvcache

__all__ = ["store_kvcache"]

