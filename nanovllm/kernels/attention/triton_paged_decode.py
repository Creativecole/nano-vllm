from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_paged_decode_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    block_tables_ptr,
    context_lens_ptr,
    out_ptr,
    SCALE: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_CHUNKS: tl.constexpr,
):
    batch_id = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)
    offs_d = tl.arange(0, HEAD_DIM)

    q_offset = batch_id * NUM_Q_HEADS * HEAD_DIM + q_head * HEAD_DIM + offs_d
    q = tl.load(q_ptr + q_offset).to(tl.float32)

    context_len = tl.load(context_lens_ptr + batch_id)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    m_i = -float("inf")
    l_i = 0.0

    offs_n_base = tl.arange(0, BLOCK_N)
    for chunk_idx in tl.range(0, MAX_CHUNKS):
        logical_pos = chunk_idx * BLOCK_N + offs_n_base
        valid = logical_pos < context_len
        logical_block = logical_pos // BLOCK_SIZE
        block_offset = logical_pos - logical_block * BLOCK_SIZE
        physical_block = tl.load(
            block_tables_ptr + batch_id * MAX_BLOCKS + logical_block,
            mask=valid & (logical_block < MAX_BLOCKS),
            other=-1,
        )
        valid = valid & (physical_block >= 0)

        kv_base = (
            physical_block[:, None] * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM
            + block_offset[:, None] * NUM_KV_HEADS * HEAD_DIM
            + kv_head * HEAD_DIM
            + offs_d[None, :]
        )
        mask = valid[:, None]
        k = tl.load(k_cache_ptr + kv_base, mask=mask, other=0.0).to(tl.float32)
        qk = tl.sum(k * q[None, :], axis=1) * SCALE
        qk = tl.where(valid, qk, -float("inf"))

        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new)
        p = tl.where(valid, p, 0.0)

        v = tl.load(v_cache_ptr + kv_base, mask=mask, other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i
    out_offset = batch_id * NUM_Q_HEADS * HEAD_DIM + q_head * HEAD_DIM + offs_d
    tl.store(out_ptr + out_offset, out)


def triton_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    block_size: int = 16,
    block_n: int | None = None,
) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("triton_paged_decode requires CUDA")
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_q_heads, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    batch_size, num_q_heads, head_dim = q.shape
    if head_dim != 128:
        raise NotImplementedError("triton_paged_decode currently supports head_dim=128")
    if k_cache.shape[1] != block_size:
        raise ValueError("block_size does not match cache layout")
    num_kv_heads = k_cache.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    block_tables = block_tables.to(device=q.device, dtype=torch.int32).contiguous()
    context_lens = context_lens.to(device=q.device, dtype=torch.int32).contiguous()
    out = torch.empty_like(q)
    scale = (head_dim ** -0.5) if scale is None else scale
    max_blocks = block_tables.shape[1]
    block_n = block_n or 32
    max_chunks = triton.cdiv(max_blocks * block_size, block_n)
    _triton_paged_decode_kernel[(batch_size, num_q_heads)](
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        out,
        SCALE=float(scale),
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        MAX_BLOCKS=max_blocks,
        BLOCK_N=block_n,
        MAX_CHUNKS=max_chunks,
        num_warps=4,
    )
    return out

