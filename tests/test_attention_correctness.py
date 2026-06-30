import pytest
import torch

from nanovllm.backends import paged_attention_decode
from nanovllm.kernels.attention import (
    build_block_tables,
    pack_dense_kv_to_cache,
    torch_paged_attention_decode,
)


def make_inputs(device):
    batch_size, seq_len, block_size = 2, 17, 4
    num_q_heads, num_kv_heads, head_dim = 4, 2, 128
    q = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=torch.float16)
    dense_k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    dense_v = torch.randn_like(dense_k)
    block_tables = build_block_tables(batch_size, seq_len, block_size, device=device)
    context_lens = torch.tensor([17, 13], device=device, dtype=torch.int32)
    k_cache, v_cache = pack_dense_kv_to_cache(dense_k, dense_v, block_tables, block_size)
    return q, k_cache, v_cache, block_tables, context_lens, block_size


def test_torch_paged_attention_decode_is_self_consistent_cpu():
    q, k_cache, v_cache, block_tables, context_lens, block_size = make_inputs("cpu")
    out = torch_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton paged attention")
def test_triton_paged_attention_decode_matches_torch_reference():
    pytest.importorskip("triton")
    q, k_cache, v_cache, block_tables, context_lens, block_size = make_inputs("cuda")
    ref = torch_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    out = paged_attention_decode(
        "triton_paged_decode",
        q,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_size=block_size,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)

