# Attention Backend

This fork exposes attention backends as standalone benchmarkable paths and as explicit eager-mode
runtime choices. The default generation runtime remains FlashAttention.

## Backends

| backend | phase | status |
|---|---|---|
| `torch_sdpa` | prefill | PyTorch reference path |
| `flash_attn` | prefill/runtime | stable nano-vLLM path, optional prefill benchmark |
| `torch_paged` | decode | correctness reference over paged KV cache |
| `triton_paged_decode` | decode | custom decode-only Triton PagedAttention v1 kernel |
| `triton_paged_decode_v2` | decode | GQA-grouped Triton PagedAttention v2 kernel |

The E2E `LLM.generate` path defaults to `flash_attn`. Custom decode backends can be selected with
`attn_backend="torch_paged"`, `attn_backend="triton_paged_decode"`, or
`attn_backend="triton_paged_decode_v2"` when `enforce_eager=True`.
CUDA Graph capture for custom decode backends is not enabled yet.

## Current API

Runtime selection:

```python
from nanovllm import LLM

llm = LLM(
    "../models/Qwen3-4B",
    enforce_eager=True,
    attn_backend="triton_paged_decode_v2",
)
```

Standalone backend selection:

```python
from nanovllm.backends import paged_attention_decode

out = paged_attention_decode(
    "triton_paged_decode_v2",
    q,
    k_cache,
    v_cache,
    block_tables,
    context_lens,
    block_size=16,
)
```

`torch_paged` is the reference path and should be used for correctness checks.

## Limitations

- Triton paged decode backends are decode-only.
- `head_dim=128` is the first supported target.
- Prefill currently compares Torch SDPA and FlashAttention; a Triton flash-style prefill kernel is TODO.
- E2E custom backend integration is explicit and eager-only; CUDA Graph support remains TODO.
