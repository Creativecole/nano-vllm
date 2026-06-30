import pytest
import torch

from nanovllm.backends import paged_attention_decode
from nanovllm.kernels.attention import build_block_tables, pack_dense_kv_to_cache, torch_paged_attention_decode


def make_decode_case(block_size: int, dtype: torch.dtype, context_lens: list[int]):
    batch_size = len(context_lens)
    seq_len = max(context_lens)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    device = torch.device("cuda")
    q = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)
    dense_k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    dense_v = torch.randn_like(dense_k)
    block_tables = build_block_tables(batch_size, seq_len, block_size, device=device)
    context_lens_tensor = torch.tensor(context_lens, device=device, dtype=torch.int32)
    k_cache, v_cache = pack_dense_kv_to_cache(dense_k, dense_v, block_tables, block_size)
    return q, k_cache, v_cache, block_tables, context_lens_tensor, block_size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton paged attention")
@pytest.mark.parametrize("block_size", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_paged_decode_v2_matches_reference_block_size_sweep(block_size, dtype):
    pytest.importorskip("triton")
    q, k_cache, v_cache, block_tables, context_lens, block_size = make_decode_case(
        block_size,
        dtype,
        context_lens=[17, 513],
    )
    ref = torch_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    out = paged_attention_decode(
        "triton_paged_decode_v2",
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=3e-2, atol=3e-2)
