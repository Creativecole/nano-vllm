from benchmarks.qwen35_hybrid.diagnose_batched_prefill import (
    parse_cases,
    summarize_case,
)


def test_parse_batched_prefill_cases():
    assert parse_cases("1x128, 4x512") == [(1, 128), (4, 512)]


def test_diagnosis_accepts_one_batched_call_per_layer():
    call = {
        "backend": "chunked",
        "is_prefill": True,
        "query_shape": [4, 128, 32, 128],
    }
    diagnostics = {
        "layers": {
            0: {
                "calls": [call],
                "recurrence_calls": 1,
                "equal_length_batched_prefill_calls": 1,
                "variable_length_fallback_calls": 0,
                "fallback_sequences": 0,
            },
            1: {
                "calls": [call],
                "recurrence_calls": 1,
                "equal_length_batched_prefill_calls": 1,
                "variable_length_fallback_calls": 0,
                "fallback_sequences": 0,
            },
        },
        "state_manager": {
            "gather_calls": 1,
            "gather_layer_ops": 2,
            "gather_bytes": 1024,
            "commit_calls": 1,
            "commit_layer_ops": 2,
            "commit_bytes": 1024,
            "slot_id_upload_bytes": 32,
        },
    }

    row = summarize_case(4, 128, 0.1, 1.0, diagnostics)

    assert row["all_layers_single_batched_call"] is True
    assert row["observed_recurrence_batch_dimensions"] == [4]
    assert row["recurrence_calls_per_layer_min"] == 1
    assert row["recurrence_calls_per_layer_max"] == 1
