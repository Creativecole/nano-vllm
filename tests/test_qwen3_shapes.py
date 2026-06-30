from types import SimpleNamespace

from nanovllm.utils.shapes import extract_qwen_attention_shapes


def test_extract_qwen3_attention_shapes_from_config_like_object():
    config = SimpleNamespace(
        vocab_size=151936,
        hidden_size=2560,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=9728,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        attention_bias=False,
        head_dim=128,
        max_position_embeddings=32768,
        dtype="bfloat16",
        tie_word_embeddings=False,
    )

    shapes = extract_qwen_attention_shapes(config, "dummy-qwen3-4b")

    assert shapes.hidden_size == 2560
    assert shapes.head_dim == 128
    assert shapes.gqa_ratio == 4
    assert shapes.q_proj_shape == (4096, 2560)
    assert shapes.k_proj_shape == (1024, 2560)
    assert shapes.v_proj_shape == (1024, 2560)
    assert shapes.qkv_proj_shape == (6144, 2560)
    assert shapes.decode_attention_shape(batch_size=2, seq_len=1024, block_size=16)["block_tables"] == [2, 64]

