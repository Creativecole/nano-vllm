from functools import lru_cache
import torch
from torch import nn
import triton
import triton.language as tl


def _next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@triton.jit
def _rotary_embedding_kernel(
    qk_ptr,         # [N, num_heads, head_dim]  — q or k
    cos_ptr,         # [N, 1, half_dim]  — cos values per position
    sin_ptr,         # [N, 1, half_dim]  — sin values per position
    stride_qk_n,    # stride along token dim
    stride_qk_h,    # stride along head dim
    stride_cs_n,    # stride along token dim for cos/sin
    HALF_DIM: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    """In-place rotary embedding for one (token, head) pair."""
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < HALF_DIM

    # load x1 (first half) and x2 (second half) of head
    base = token_id * stride_qk_n + head_id * stride_qk_h
    x1 = tl.load(qk_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(qk_ptr + base + HALF_DIM + cols, mask=mask, other=0.0).to(tl.float32)

    # load cos, sin for this token  (broadcast across heads via stride_cs_n)
    cs_base = token_id * stride_cs_n
    cos = tl.load(cos_ptr + cs_base + cols, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_base + cols, mask=mask, other=0.0).to(tl.float32)

    # apply rotation in-place
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin

    tl.store(qk_ptr + base + cols, y1.to(qk_ptr.dtype.element_ty), mask=mask)
    tl.store(qk_ptr + base + HALF_DIM + cols, y2.to(qk_ptr.dtype.element_ty), mask=mask)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.half_dim = head_size // 2
        self.block_hd = _next_power_of_2(self.half_dim)
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos().unsqueeze_(1)   # [max_pos, 1, half_dim]
        sin = freqs.sin().unsqueeze_(1)   # [max_pos, 1, half_dim]
        self.register_buffer("cos_cache", cos, persistent=False)
        self.register_buffer("sin_cache", sin, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cache[positions]   # [N, 1, half_dim]
        sin = self.sin_cache[positions]   # [N, 1, half_dim]

        N = query.shape[0]
        num_q_heads = query.shape[1]
        num_k_heads = key.shape[1]

        stride_cs_n = cos.stride(0)

        # in-place rotary on query: grid = (N_tokens, num_q_heads)
        _rotary_embedding_kernel[(N, num_q_heads)](
            query, cos, sin,
            query.stride(0), query.stride(1), stride_cs_n,
            HALF_DIM=self.half_dim, BLOCK_HD=self.block_hd,
        )
        # in-place rotary on key: grid = (N_tokens, num_k_heads)
        _rotary_embedding_kernel[(N, num_k_heads)](
            key, cos, sin,
            key.stride(0), key.stride(1), stride_cs_n,
            HALF_DIM=self.half_dim, BLOCK_HD=self.block_hd,
        )
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
