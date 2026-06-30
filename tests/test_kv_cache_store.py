import pytest
import torch


pytest.importorskip("triton")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton KV-cache store tests", allow_module_level=True)


def test_store_kvcache_matches_reference():
    from nanovllm.kernels.kv_cache import store_kvcache

    n_tokens, num_heads, head_dim = 4, 2, 8
    key = torch.randn(n_tokens, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    k_cache = torch.zeros(8, 16, num_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.tensor([3, 17, -1, 25], device="cuda", dtype=torch.int32)

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize()

    k_ref = torch.zeros_like(k_cache).view(-1, num_heads, head_dim)
    v_ref = torch.zeros_like(v_cache).view(-1, num_heads, head_dim)
    for token_id, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot >= 0:
            k_ref[slot] = key[token_id]
            v_ref[slot] = value[token_id]

    torch.testing.assert_close(k_cache.view_as(k_ref), k_ref)
    torch.testing.assert_close(v_cache.view_as(v_ref), v_ref)

