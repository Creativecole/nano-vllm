from __future__ import annotations

import torch
import torch.nn.functional as F


def torch_sdpa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Reference varlen causal prefill attention using PyTorch SDPA.

    q: [total_tokens, num_q_heads, head_dim]
    k/v: [total_tokens, num_kv_heads, head_dim]
    cu_seqlens: [num_seqs + 1]
    """
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q/k/v must be [total_tokens, heads, head_dim]")
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    scale = (q.shape[-1] ** -0.5) if scale is None else scale
    outputs = []
    cu_cpu = cu_seqlens.detach().cpu().tolist()
    for start, end in zip(cu_cpu, cu_cpu[1:]):
        q_seq = q[start:end].transpose(0, 1).unsqueeze(0)
        k_seq = k[start:end].transpose(0, 1).unsqueeze(0)
        v_seq = v[start:end].transpose(0, 1).unsqueeze(0)
        if num_q_heads != num_kv_heads:
            repeat = num_q_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(repeat, dim=1)
            v_seq = v_seq.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
            scale=scale,
        )
        outputs.append(out.squeeze(0).transpose(0, 1))
    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)

