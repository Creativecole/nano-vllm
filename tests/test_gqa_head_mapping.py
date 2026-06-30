import pytest

from nanovllm.kernels.attention import gqa_kv_head


def test_gqa_head_mapping_for_qwen_style_heads():
    assert [gqa_kv_head(i, 16, 4) for i in range(16)] == [
        0, 0, 0, 0,
        1, 1, 1, 1,
        2, 2, 2, 2,
        3, 3, 3, 3,
    ]


def test_gqa_head_mapping_rejects_invalid_ratio():
    with pytest.raises(ValueError):
        gqa_kv_head(0, 10, 4)


def test_gqa_head_mapping_rejects_out_of_range_head():
    with pytest.raises(ValueError):
        gqa_kv_head(16, 16, 4)

