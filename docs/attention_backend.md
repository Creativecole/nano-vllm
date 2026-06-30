# Attention Backend

This fork exposes attention backends as standalone benchmarkable paths before changing the default
generation runtime.

## Backends

| backend | phase | status |
|---|---|---|
| `torch_sdpa` | prefill | PyTorch reference path |
| `flash_attn` | prefill/runtime | stable nano-vLLM path, optional prefill benchmark |
| `torch_paged` | decode | correctness reference over paged KV cache |
| `triton_paged_decode` | decode | custom decode-only Triton PagedAttention kernel |

The E2E `LLM.generate` path still uses `flash_attn`. Custom decode backends should first pass
correctness and standalone latency benchmarks, then be wired into `ModelRunner` with CUDA Graph
handling.

## Current API

```python
from nanovllm.backends import paged_attention_decode

out = paged_attention_decode(
    "triton_paged_decode",
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

- `triton_paged_decode` is decode-only.
- `head_dim=128` is the first supported target.
- Prefill currently compares Torch SDPA and FlashAttention; a Triton flash-style prefill kernel is TODO.
- E2E custom backend integration is not enabled by default.

