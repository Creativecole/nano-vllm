import torch
from torch import nn
import triton
import triton.language as tl
from inspect import signature

try:
    from flash_attn_interface import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


def _next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


# ============================================================
# FP8 quantization helpers
# ============================================================
FP8_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)
FP8_E4M3_MAX = torch.finfo(FP8_E4M3_DTYPE).max if FP8_E4M3_DTYPE is not None else 448.0


try:
    _FLASH_ATTN_KVCACHE_PARAMS = set(signature(flash_attn_with_kvcache).parameters)
except (TypeError, ValueError):
    _FLASH_ATTN_KVCACHE_PARAMS = set()


def flash_attn_supports_fp8_kvcache() -> bool:
    return "k_descale" in _FLASH_ATTN_KVCACHE_PARAMS and "v_descale" in _FLASH_ATTN_KVCACHE_PARAMS


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


@triton.jit
def store_kvcache_fp8_kernel(
    key_ptr,            # [N, num_kv_heads, head_dim]  BF16
    value_ptr,          # [N, num_kv_heads, head_dim]  BF16
    k_cache_ptr,        # [num_blocks * block_size, num_kv_heads, head_dim]  FP8
    v_cache_ptr,        # [num_blocks * block_size, num_kv_heads, head_dim]  FP8
    k_scale_ptr,        # [num_layers, num_kv_heads]  or scalar  FP32
    v_scale_ptr,        # [num_layers, num_kv_heads]  or scalar  FP32
    slot_mapping_ptr,
    stride_kv_n,
    stride_kv_h,
    stride_cache_s,
    stride_cache_h,
    stride_scale_layer,  # stride for scale along layer dim
    layer_id,
    HEAD_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    """2D grid: (N_tokens, num_kv_heads). Fused BF16→FP8 quantization + cache write."""
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + token_id)
    if slot == -1:
        return

    cols = tl.arange(0, BLOCK_HD)
    mask = cols < HEAD_DIM

    # load BF16 key/value
    kv_offset = token_id * stride_kv_n + head_id * stride_kv_h + cols
    key = tl.load(key_ptr + kv_offset, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(value_ptr + kv_offset, mask=mask, other=0.0).to(tl.float32)

    # load per-head scale factors
    scale_offset = layer_id * stride_scale_layer + head_id
    k_scale = tl.load(k_scale_ptr + scale_offset)
    v_scale = tl.load(v_scale_ptr + scale_offset)

    # quantize: fp8_val = clamp(bf16_val / scale, -FP8_MAX, FP8_MAX)
    key_fp8 = tl.clamp(key / k_scale, -FP8_MAX, FP8_MAX).to(k_cache_ptr.dtype.element_ty)
    value_fp8 = tl.clamp(value / v_scale, -FP8_MAX, FP8_MAX).to(v_cache_ptr.dtype.element_ty)

    # store FP8 to cache
    cache_offset = slot * stride_cache_s + head_id * stride_cache_h + cols
    tl.store(k_cache_ptr + cache_offset, key_fp8, mask=mask)
    tl.store(v_cache_ptr + cache_offset, value_fp8, mask=mask)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """Standard BF16 KV cache store (non-FP8 path)."""
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


def store_kvcache_fp8(
    key: torch.Tensor, value: torch.Tensor,
    k_cache: torch.Tensor, v_cache: torch.Tensor,
    k_scale: torch.Tensor, v_scale: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_id: int,
):
    """FP8 KV cache store with fused quantization."""
    N, num_kv_heads, head_dim = key.shape
    BLOCK_HD = _next_power_of_2(head_dim)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    store_kvcache_fp8_kernel[(N, num_kv_heads)](
        key, value, k_cache, v_cache,
        k_scale, v_scale,
        slot_mapping,
        key.stride(0), key.stride(1),
        k_cache.stride(1), k_cache.stride(2),
        k_scale.stride(0),
        layer_id,
        HEAD_DIM=head_dim, BLOCK_HD=BLOCK_HD,
        FP8_MAX=FP8_E4M3_MAX,
    )


def _call_flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    context,
    softmax_scale: float,
    fp8_kv: bool,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
    layer_id: int,
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

    if fp8_kv:
        if k_scale is None or v_scale is None:
            raise RuntimeError("FP8 KV cache requires k_scale and v_scale tensors.")
        if not flash_attn_supports_fp8_kvcache():
            raise RuntimeError(
                "FP8 KV cache requires a FlashAttention build whose "
                "flash_attn_with_kvcache exposes k_descale/v_descale."
            )
        kwargs["k_descale"] = k_scale[layer_id] if k_scale.ndim == 2 else k_scale
        kwargs["v_descale"] = v_scale[layer_id] if v_scale.ndim == 2 else v_scale
        if "q_descale" in _FLASH_ATTN_KVCACHE_PARAMS:
            kwargs["q_descale"] = None

    return flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, **kwargs)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        layer_id: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.layer_id = layer_id
        self.k_cache = self.v_cache = torch.tensor([])
        # FP8 scale factors — set by model_runner when fp8_kv is enabled
        self.k_scale = None
        self.v_scale = None
        self.cache_dtype = "bf16"
        self.fp8_kv = False

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            if self.fp8_kv:
                store_kvcache_fp8(
                    k, v, k_cache, v_cache,
                    self.k_scale, self.v_scale,
                    context.slot_mapping, self.layer_id,
                )
            else:
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                if self.fp8_kv:
                    raise RuntimeError(
                        "FP8 KV cache does not support prefix-cache prefill in "
                        "the current nano-vLLM path. Use kv_cache_dtype='bf16' "
                        "for workloads that reuse cached prompt blocks."
                    )
                else:
                    k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = _call_flash_attn_with_kvcache(
                q, k_cache, v_cache, context, self.scale,
                self.fp8_kv, self.k_scale, self.v_scale, self.layer_id,
            )
        return o
