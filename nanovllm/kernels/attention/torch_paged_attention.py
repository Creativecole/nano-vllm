from __future__ import annotations

import torch


def gqa_kv_head(q_head: int, num_q_heads: int, num_kv_heads: int) -> int:
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"num_q_heads={num_q_heads} must be divisible by num_kv_heads={num_kv_heads}")
    if q_head < 0 or q_head >= num_q_heads:
        raise ValueError(f"q_head={q_head} is outside [0, {num_q_heads})")
    return q_head // (num_q_heads // num_kv_heads)


def build_block_tables(batch_size: int, seq_len: int, block_size: int, device=None) -> torch.Tensor:
    max_blocks = (seq_len + block_size - 1) // block_size
    tables = torch.empty(batch_size, max_blocks, dtype=torch.int32, device=device)
    for batch_id in range(batch_size):
        start = batch_id * max_blocks
        tables[batch_id] = torch.arange(start, start + max_blocks, dtype=torch.int32, device=device)
    return tables


def pack_dense_kv_to_cache(
    dense_k: torch.Tensor,
    dense_v: torch.Tensor,
    block_tables: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack dense KV tensors into a paged cache using the supplied block tables.

    dense_k/dense_v: [batch, seq_len, num_kv_heads, head_dim]
    block_tables: [batch, max_blocks_per_seq]
    returns: [num_physical_blocks, block_size, num_kv_heads, head_dim]
    """
    if dense_k.shape != dense_v.shape:
        raise ValueError("dense_k and dense_v must have the same shape")
    if dense_k.ndim != 4:
        raise ValueError("dense_k must have shape [batch, seq_len, num_kv_heads, head_dim]")
    batch_size, seq_len, num_kv_heads, head_dim = dense_k.shape
    if block_tables.shape[0] != batch_size:
        raise ValueError("block_tables batch dimension must match dense KV batch")
    max_block_id = int(block_tables.max().item()) if block_tables.numel() else -1
    k_cache = torch.zeros(
        max_block_id + 1,
        block_size,
        num_kv_heads,
        head_dim,
        device=dense_k.device,
        dtype=dense_k.dtype,
    )
    v_cache = torch.zeros_like(k_cache)
    for batch_id in range(batch_size):
        for pos in range(seq_len):
            block_idx = pos // block_size
            block_offset = pos % block_size
            physical_block = int(block_tables[batch_id, block_idx].item())
            k_cache[physical_block, block_offset] = dense_k[batch_id, pos]
            v_cache[physical_block, block_offset] = dense_v[batch_id, pos]
    return k_cache, v_cache


def _gather_sequence_cache(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    context_len: int,
    block_size: int,
) -> torch.Tensor:
    num_kv_heads = cache.shape[2]
    head_dim = cache.shape[3]
    out = torch.empty(context_len, num_kv_heads, head_dim, device=cache.device, dtype=cache.dtype)
    for pos in range(context_len):
        block_idx = pos // block_size
        block_offset = pos % block_size
        physical_block = int(block_table[block_idx].item())
        if physical_block < 0:
            raise ValueError(f"block_table contains invalid block id at logical block {block_idx}")
        out[pos] = cache[physical_block, block_offset]
    return out


def torch_paged_attention_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    block_size: int = 16,
) -> torch.Tensor:
    """Reference decode attention over a paged KV cache.

    q: [batch, num_q_heads, head_dim]
    cache: [num_blocks, block_size, num_kv_heads, head_dim]
    block_tables: [batch, max_blocks_per_seq]
    context_lens: [batch]
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [batch, num_q_heads, head_dim]")
    if k_cache.shape != v_cache.shape:
        raise ValueError("k_cache and v_cache must have the same shape")
    batch_size, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    if k_cache.shape[1] != block_size:
        raise ValueError("block_size does not match cache layout")
    if block_tables.shape[0] != batch_size or context_lens.shape[0] != batch_size:
        raise ValueError("batch dimensions do not match")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    scale = (head_dim ** -0.5) if scale is None else scale
    out = torch.empty_like(q)
    for batch_id in range(batch_size):
        context_len = int(context_lens[batch_id].item())
        if context_len <= 0:
            out[batch_id].zero_()
            continue
        k_seq = _gather_sequence_cache(k_cache, block_tables[batch_id], context_len, block_size).float()
        v_seq = _gather_sequence_cache(v_cache, block_tables[batch_id], context_len, block_size).float()
        for q_head in range(num_q_heads):
            kv_head = gqa_kv_head(q_head, num_q_heads, num_kv_heads)
            scores = torch.matmul(k_seq[:, kv_head, :], q[batch_id, q_head].float()) * scale
            probs = torch.softmax(scores, dim=-1)
            out[batch_id, q_head] = torch.matmul(probs, v_seq[:, kv_head, :]).to(out.dtype)
    return out

