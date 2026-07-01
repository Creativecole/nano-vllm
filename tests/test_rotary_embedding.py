import pytest
import torch

pytest.importorskip("triton")

from nanovllm.layers.rotary_embedding import RotaryEmbedding


def torch_rope_reference(rope: RotaryEmbedding, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor):
    cos = rope.cos_cache[positions]
    sin = rope.sin_cache[positions]

    def apply(x):
        half = x.shape[-1] // 2
        x1 = x[..., :half].float()
        x2 = x[..., half:].float()
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat([y1, y2], dim=-1).to(x.dtype)

    return apply(query), apply(key)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Triton RoPE")
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_triton_rope_matches_indexed_reference(batch_size, dtype):
    device = torch.device("cuda")
    head_dim = 128
    num_q_heads = 4
    num_k_heads = 2
    max_position = 4096
    torch.manual_seed(0)

    rope = RotaryEmbedding(
        head_size=head_dim,
        rotary_dim=head_dim,
        max_position_embeddings=max_position,
        base=1000000,
    ).to(device=device)
    positions = torch.randint(0, max_position, (batch_size,), device=device, dtype=torch.int64)
    query = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_k_heads, head_dim, device=device, dtype=dtype)

    expected_q, expected_k = torch_rope_reference(rope, positions, query, key)
    actual_q, actual_k = rope(positions, query.clone(), key.clone())
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_q.float(), expected_q.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_k.float(), expected_k.float(), rtol=2e-2, atol=2e-2)
