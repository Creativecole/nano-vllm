import torch
from torch import nn
import triton
import triton.language as tl
from inspect import signature

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.kernels.attention.torch_paged_attention import torch_paged_attention_decode
from nanovllm.kernels.attention.triton_paged_decode import triton_paged_attention_decode
from nanovllm.kernels.attention.triton_paged_decode_v2 import triton_paged_attention_decode_v2
from nanovllm.utils.context import get_context


def _next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


try:
    _FLASH_ATTN_KVCACHE_PARAMS = set(signature(flash_attn_with_kvcache).parameters)
except (TypeError, ValueError):
    _FLASH_ATTN_KVCACHE_PARAMS = set()


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """1D grid: each program stores one token's full BF16 K/V vector."""
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1:
        return
    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


@triton.jit
def store_kvcache_2d_kernel(
    key_ptr,
    value_ptr,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    stride_kv_n,      # stride along token dim for key/value
    stride_kv_h,      # stride along head dim for key/value  (= head_dim)
    stride_cache_s,    # stride along slot dim for cache  (= num_kv_heads * head_dim)
    stride_cache_h,    # stride along head dim for cache  (= head_dim)
    HEAD_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """2D grid: (N_tokens, num_kv_heads). Each program stores one head."""
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + token_id)
    if slot == -1:
        return

    cols = tl.arange(0, BLOCK_HD)
    mask = cols < HEAD_DIM

    # key: [N, num_kv_heads, head_dim]
    kv_offset = token_id * stride_kv_n + head_id * stride_kv_h + cols
    key = tl.load(key_ptr + kv_offset, mask=mask)
    value = tl.load(value_ptr + kv_offset, mask=mask)

    # cache: [num_blocks * block_size, num_kv_heads, head_dim]
    cache_offset = slot * stride_cache_s + head_id * stride_cache_h + cols
    tl.store(k_cache_ptr + cache_offset, key, mask=mask)
    tl.store(v_cache_ptr + cache_offset, value, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Store K/V tensors into the paged KV cache."""
    N, num_kv_heads, head_dim = key.shape
    D = num_kv_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_2d(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Experimental 2D BF16 KV cache store used by microbenchmarks."""
    N, num_kv_heads, head_dim = key.shape
    BLOCK_HD = _next_power_of_2(head_dim)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    store_kvcache_2d_kernel[(N, num_kv_heads)](
        key, value, k_cache, v_cache, slot_mapping,
        key.stride(0), key.stride(1),
        k_cache.stride(1), k_cache.stride(2),
        HEAD_DIM=head_dim, BLOCK_HD=BLOCK_HD,
    )


def _call_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    softmax_scale: float,
):
    kwargs = dict(
        cache_seqlens=context.context_lens,
        softmax_scale=softmax_scale,
        causal=True,
    )
    if "page_table" in _FLASH_ATTN_KVCACHE_PARAMS:
        kwargs["page_table"] = context.block_tables
    else:
        kwargs["block_table"] = context.block_tables

    return flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, **kwargs)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        layer_id: int = 0,
        attn_backend: str = "flash_attn",
        block_size: int = 256,
        triton_paged_decode_auto_threshold: int = 1024,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.attn_backend = attn_backend
        self.block_size = block_size
        self.triton_paged_decode_auto_threshold = triton_paged_decode_auto_threshold
        self.k_cache = self.v_cache = torch.tensor([])

    def _resolve_decode_backend(self, context) -> str:
        if self.attn_backend != "triton_paged_decode_auto":
            return self.attn_backend
        if context.max_context_len < self.triton_paged_decode_auto_threshold:
            return "triton_paged_decode"
        return "triton_paged_decode_v2"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            decode_backend = self._resolve_decode_backend(context)
            if decode_backend == "flash_attn":
                o = _call_flash_attn_with_kvcache(
                    q, k_cache, v_cache, context, self.scale,
                )
            elif decode_backend == "triton_paged_decode":
                o = triton_paged_attention_decode(
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.context_lens,
                    scale=self.scale,
                    block_size=self.block_size,
                )
            elif decode_backend == "triton_paged_decode_v2":
                o = triton_paged_attention_decode_v2(
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.context_lens,
                    scale=self.scale,
                    block_size=self.block_size,
                )
            elif decode_backend == "torch_paged":
                o = torch_paged_attention_decode(
                    q,
                    k_cache,
                    v_cache,
                    context.block_tables,
                    context.context_lens,
                    scale=self.scale,
                    block_size=self.block_size,
                )
            else:
                raise RuntimeError(f"Unsupported decode attention backend: {self.attn_backend}")
        return o
