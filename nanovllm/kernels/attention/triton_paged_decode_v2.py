from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _triton_paged_decode_v2_kernel(
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
    GQA_GROUP: tl.constexpr,
):
    batch_id = tl.program_id(0)
    kv_head = tl.program_id(1)

    offs_g = tl.arange(0, GQA_GROUP)
    offs_d = tl.arange(0, HEAD_DIM)
    q_heads = kv_head * GQA_GROUP + offs_g
    q_offsets = batch_id * NUM_Q_HEADS * HEAD_DIM + q_heads[:, None] * HEAD_DIM + offs_d[None, :]
    q = tl.load(q_ptr + q_offsets)

    context_len = tl.load(context_lens_ptr + batch_id)
    acc = tl.zeros((GQA_GROUP, HEAD_DIM), dtype=tl.float32)
    m_i = tl.full((GQA_GROUP,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((GQA_GROUP,), dtype=tl.float32)

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

        kv_offsets = (
            physical_block[:, None] * BLOCK_SIZE * NUM_KV_HEADS * HEAD_DIM
            + block_offset[:, None] * NUM_KV_HEADS * HEAD_DIM
            + kv_head * HEAD_DIM
            + offs_d[None, :]
        )
        mask = valid[:, None]
        k = tl.load(k_cache_ptr + kv_offsets, mask=mask, other=0.0)
        qk = tl.dot(k, tl.trans(q), input_precision="tf32") * SCALE
        qk = tl.where(valid[:, None], qk, -float("inf"))

        m_ij = tl.max(qk, axis=0)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[None, :])
        p = tl.where(valid[:, None], p, 0.0)

        v = tl.load(v_cache_ptr + kv_offsets, mask=mask, other=0.0)
        acc = acc * alpha[:, None] + tl.dot(tl.trans(p), v.to(tl.float32), input_precision="tf32")
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out = acc / l_i[:, None]
    out_offsets = batch_id * NUM_Q_HEADS * HEAD_DIM + q_heads[:, None] * HEAD_DIM + offs_d[None, :]
    tl.store(out_ptr + out_offsets, out)


def _contiguous_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _int32_cuda_contiguous(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device != device or tensor.dtype != torch.int32:
        tensor = tensor.to(device=device, dtype=torch.int32)
    return _contiguous_if_needed(tensor)


def _default_block_n(block_size: int, max_context_tokens: int, gqa_group: int) -> int:
    # Grouping several Q heads in one program increases register pressure.
    # Keep the tile conservative for long contexts and larger GQA groups.
    if gqa_group >= 4:
        return 32
    if max_context_tokens <= 4096 and block_size <= 64:
        return 64
    return 32


def triton_paged_attention_decode_v2(
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
        raise RuntimeError("triton_paged_decode_v2 requires CUDA")
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_q_heads, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    batch_size, num_q_heads, head_dim = q.shape
    if head_dim != 128:
        raise NotImplementedError("triton_paged_decode_v2 currently supports head_dim=128")
    if k_cache.shape[1] != block_size:
        raise ValueError("block_size does not match cache layout")
    num_kv_heads = k_cache.shape[2]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    gqa_group = num_q_heads // num_kv_heads
    if gqa_group not in {1, 2, 4, 8}:
        raise NotImplementedError("triton_paged_decode_v2 supports GQA groups 1, 2, 4, or 8")

    q = _contiguous_if_needed(q)
    k_cache = _contiguous_if_needed(k_cache)
    v_cache = _contiguous_if_needed(v_cache)
    block_tables = _int32_cuda_contiguous(block_tables, q.device)
    context_lens = _int32_cuda_contiguous(context_lens, q.device)
    out = torch.empty_like(q)

    scale = (head_dim ** -0.5) if scale is None else scale
    max_blocks = block_tables.shape[1]
    max_context_tokens = max_blocks * block_size
    block_n = block_n or _default_block_n(block_size, max_context_tokens, gqa_group)
    max_chunks = triton.cdiv(max_context_tokens, block_n)
    _triton_paged_decode_v2_kernel[(batch_size, num_kv_heads)](
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
        GQA_GROUP=gqa_group,
        num_warps=4,
        num_stages=3,
    )
    return out
