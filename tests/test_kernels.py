import pytest
import torch


pytest.importorskip("triton")
pytest.importorskip("flash_attn")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton kernel tests", allow_module_level=True)


def test_store_kvcache_matches_reference():
    from nanovllm.layers.attention import store_kvcache

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
    flat_slots = slot_mapping.cpu().tolist()
    for token_id, slot in enumerate(flat_slots):
        if slot >= 0:
            k_ref[slot] = key[token_id]
            v_ref[slot] = value[token_id]

    torch.testing.assert_close(k_cache.view_as(k_ref), k_ref)
    torch.testing.assert_close(v_cache.view_as(v_ref), v_ref)


def test_silu_and_mul_matches_torch():
    from nanovllm.layers.activation import SiluAndMul

    x = torch.randn(7, 64, device="cuda", dtype=torch.bfloat16)
    out = SiluAndMul()(x)
    gate, up = x.chunk(2, -1)
    ref = torch.nn.functional.silu(gate.float()) * up.float()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_rms_norm_matches_torch():
    from nanovllm.layers.layernorm import RMSNorm

    x = torch.randn(5, 32, device="cuda", dtype=torch.bfloat16)
    norm = RMSNorm(32).cuda()
    out = norm(x)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + norm.eps)
    ref = ref * norm.weight.float()
    torch.testing.assert_close(out.float(), ref, rtol=2e-2, atol=2e-2)


def test_rotary_embedding_matches_reference():
    from nanovllm.layers.rotary_embedding import RotaryEmbedding

    rope = RotaryEmbedding(16, 16, 128, 10000).cuda()
    positions = torch.tensor([0, 1, 7], device="cuda", dtype=torch.int64)
    q = torch.randn(3, 2, 16, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 1, 16, device="cuda", dtype=torch.bfloat16)
    q_in, k_in = q.clone(), k.clone()

    q_out, k_out = rope(positions, q, k)
    cos = rope.cos_cache[positions].cuda()
    sin = rope.sin_cache[positions].cuda()

    def ref_rotate(x):
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)

    torch.testing.assert_close(q_out.float(), ref_rotate(q_in), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_out.float(), ref_rotate(k_in), rtol=2e-2, atol=2e-2)
