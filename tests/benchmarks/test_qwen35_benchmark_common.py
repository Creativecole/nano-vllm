import pytest
import torch

from benchmarks.qwen35_hybrid.bench_e2e import aggregate_rows
from benchmarks.qwen35_hybrid.common import (
    parse_int_list,
    percentile,
    summarize,
    theoretical_cache_bytes,
)


def test_parse_and_percentile_helpers():
    assert parse_int_list("1, 8,32") == [1, 8, 32]
    with pytest.raises(ValueError):
        parse_int_list("1,0")
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert summarize([1, 2, 3])["p95"] == pytest.approx(2.9)


def test_hybrid_cache_bytes_keep_kv_and_delta_separate():
    facts = {
        "dtype": str(torch.bfloat16),
        "full_attention_layers": 2,
        "linear_attention_layers": 3,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
    }
    kv_bytes, delta_bytes = theoretical_cache_bytes(facts, 2, 16)
    assert kv_bytes == 2 * 2 * 2 * 16 * 2 * 8 * torch.bfloat16.itemsize
    conv_dim = 2 * 8 * 2 + 4 * 8
    per_delta_layer = conv_dim * 4 * torch.bfloat16.itemsize
    per_delta_layer += 4 * 8 * 8 * torch.float32.itemsize
    assert delta_bytes == 3 * 2 * per_delta_layer


def test_e2e_aggregation_preserves_mean_p50_p95():
    base = {
        "backend": "nanovllm",
        "batch_size": 1,
        "prompt_len": 128,
        "output_len": 32,
        "active_kv_cache_bytes": 100,
        "active_delta_state_bytes": 200,
        "engine_kv_cache_bytes": 300,
        "engine_delta_pool_bytes": 400,
    }
    rows = []
    for repeat, elapsed in enumerate((1.0, 2.0, 3.0)):
        row = {**base, "repeat": repeat}
        for metric in (
            "elapsed_s",
            "ttft_s",
            "average_itl_ms",
            "p50_itl_ms",
            "p95_itl_ms",
            "decode_tokens_per_s",
            "e2e_tokens_per_s",
            "peak_memory_gb",
        ):
            row[metric] = elapsed
        rows.append(row)
    summary = aggregate_rows(rows)[0]
    assert summary["elapsed_s_mean"] == 2.0
    assert summary["elapsed_s_p50"] == 2.0
    assert summary["elapsed_s_p95"] == pytest.approx(2.9)
