from __future__ import annotations

import torch

from nanovllm.kernels.attention.torch_paged_attention import torch_paged_attention_decode
from nanovllm.kernels.attention.torch_attention import torch_sdpa_prefill


ATTENTION_BACKENDS = {
    "torch_sdpa",
    "flash_attn",
    "torch_paged",
    "triton_paged_decode",
}


class AttentionBackendError(RuntimeError):
    pass


def require_backend(name: str) -> None:
    if name not in ATTENTION_BACKENDS:
        raise AttentionBackendError(
            f"Unknown attention backend '{name}'. Available backends: {sorted(ATTENTION_BACKENDS)}"
        )
    if name == "flash_attn":
        try:
            import flash_attn  # noqa: F401
        except Exception as exc:  # pragma: no cover - depends on optional CUDA wheel
            raise AttentionBackendError("flash_attn backend requested, but flash-attn is not importable") from exc
    if name == "triton_paged_decode":
        try:
            import triton  # noqa: F401
        except Exception as exc:  # pragma: no cover
            raise AttentionBackendError("triton_paged_decode backend requested, but Triton is not importable") from exc
        if not torch.cuda.is_available():
            raise AttentionBackendError("triton_paged_decode backend requires CUDA")


def paged_attention_decode(
    backend: str,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lens: torch.Tensor,
    scale: float | None = None,
    block_size: int = 16,
) -> torch.Tensor:
    require_backend(backend)
    if backend == "torch_paged":
        return torch_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    if backend == "triton_paged_decode":
        from nanovllm.kernels.attention.triton_paged_decode import triton_paged_attention_decode

        return triton_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    raise AttentionBackendError(
        f"Backend '{backend}' is not a paged decode backend. "
        "Use 'torch_paged' as the reference path or 'triton_paged_decode' for the custom kernel."
    )


def prefill_attention(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    require_backend(backend)
    if backend == "torch_sdpa":
        return torch_sdpa_prefill(q, k, v, cu_seqlens, scale)
    if backend == "flash_attn":
        from flash_attn import flash_attn_varlen_func

        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            softmax_scale=scale,
            causal=True,
        )
    raise AttentionBackendError(
        f"Backend '{backend}' is not a prefill backend. Use 'torch_sdpa' or 'flash_attn'."
    )
