from argparse import Namespace

import pytest

from benchmarks.qwen35_hybrid.profile_serving import (
    derive_answers,
    kernel_category,
    load_checkpoint,
    matrix_spec,
    summarize_profile,
)
from benchmarks.qwen35_hybrid.common import write_json
from benchmarks.qwen35_hybrid.run_nsight import build_command
from benchmarks.qwen35_hybrid.profile_layers import (
    resolve_layer_ids,
    summarize_layer_profile,
)


class FakeEvent:
    def __init__(
        self,
        name,
        *,
        device_type="DeviceType.CPU",
        device_us=0.0,
        self_device_us=0.0,
        cpu_us=0.0,
        self_cpu_us=0.0,
        count=1,
    ):
        self.name = name
        self.key = name
        self.device_type = device_type
        self.device_time_total = device_us
        self.self_device_time_total = self_device_us
        self.cpu_time_total = cpu_us
        self.self_cpu_time_total = self_cpu_us
        self.count = count


class FakeProfile:
    def __init__(self, events, averages):
        self._events = events
        self._averages = averages

    def events(self):
        return self._events

    def key_averages(self):
        return self._averages


def test_kernel_categories_keep_gemm_and_state_movement_separate():
    assert kernel_category("void cutlass::gemm_kernel") == "Linear/GEMM"
    assert kernel_category("flash_fwd_splitkv_kernel") == "Full Attention"
    assert kernel_category("vectorized_gather_kernel") == "State gather/scatter"
    assert kernel_category("cudnn_conv_depthwise") == "Convolution"


def test_profile_summary_uses_cuda_leaf_events_for_additive_time():
    profile = FakeProfile(
        events=[
            FakeEvent(
                "void cutlass::gemm_kernel",
                device_type="DeviceType.CUDA",
                device_us=800.0,
            ),
            FakeEvent(
                "vectorized_gather_kernel",
                device_type="DeviceType.CUDA",
                device_us=200.0,
            ),
        ],
        averages=[
            FakeEvent(
                "qwen35_deltanet_mixer", device_us=900.0, cpu_us=1200.0, count=2
            ),
            FakeEvent(
                "cudaLaunchKernel", self_cpu_us=50.0, cpu_us=70.0, count=2
            ),
            FakeEvent("aten::copy_", self_cpu_us=20.0, cpu_us=30.0, count=4),
        ],
    )
    summary = summarize_profile(profile, wall_time_s=0.01)
    assert summary["kernel_self_cuda_total_ms"] == pytest.approx(1.0)
    assert summary["kernel_categories"][0]["category"] == "Linear/GEMM"
    assert summary["runtime_self_cpu_total_ms"] == pytest.approx(0.05)
    assert summary["operators"]["aten::copy_"]["calls"] == 4


def test_derived_answers_report_component_and_kernel_denominators():
    row = {
        "batch_size": 1,
        "prompt_len": 128,
        "decode_steps": 32,
        "phase": "decode",
        "wall_time_s": 0.1,
        "kernel_self_cuda_total_ms": 10.0,
        "runtime_self_cpu_total_ms": 2.0,
        "range_attribution": {
            "qwen35_full_attention_mixer": {"cuda_total_ms": 2.0},
            "qwen35_deltanet_mixer": {"cuda_total_ms": 6.0},
            "qwen35_mlp": {"cuda_total_ms": 2.0},
            "qwen35_deltanet_recurrence": {
                "cuda_total_ms": 3.0,
                "calls": 6,
            },
        },
        "kernel_categories": [
            {"category": "Linear/GEMM", "self_cuda_time_ms": 8.0},
            {"category": "State gather/scatter", "self_cuda_time_ms": 2.0},
        ],
        "top_kernels": [
            {"name": "gemm", "self_cuda_time_ms": 8.0},
            {"name": "gather", "self_cuda_time_ms": 2.0},
        ],
    }
    answers = derive_answers([row])
    assert answers["overall"]["full_attention_component_share"] == pytest.approx(0.2)
    assert answers["overall"]["deltanet_component_share"] == pytest.approx(0.6)
    assert answers["overall"]["linear_gemm_kernel_share"] == pytest.approx(0.8)
    assert answers["overall"]["recurrence_avg_us"] == pytest.approx(500.0)


def _nsight_args(tool, kernel_name=None):
    return Namespace(
        tool=tool,
        model="/models/Qwen3.5-9B",
        phase="decode",
        batch_size=4,
        prompt_len=512,
        decode_steps=32,
        warmup=1,
        output="/tmp/qwen35_profile",
        kernel_name=kernel_name,
        launch_skip=2,
        launch_count=1,
    )


def test_nsight_commands_target_one_reproducible_case():
    nsys = build_command(_nsight_args("nsys"))
    assert nsys[:2] == ["nsys", "profile"]
    assert "--target-only" in nsys
    ncu = build_command(_nsight_args("ncu", "gated_delta.*"))
    assert ncu[0] == "ncu"
    assert "regex:gated_delta.*" in ncu
    with pytest.raises(ValueError, match="kernel-name"):
        build_command(_nsight_args("ncu"))


def _profile_args(path):
    return Namespace(
        model="/models/Qwen3.5-9B",
        batch_sizes="1,4",
        prompt_lens="128",
        decode_steps="32",
        phases="prefill,decode",
        warmup=1,
        record_shapes=True,
        profile_memory=False,
        no_resume=False,
        save_json=str(path),
    )


def test_profile_checkpoint_resumes_only_an_identical_matrix(tmp_path):
    path = tmp_path / "profile.json"
    args = _profile_args(path)
    spec = matrix_spec(args)
    row = {
        "batch_size": 1,
        "prompt_len": 128,
        "decode_steps": 32,
        "phase": "prefill",
    }
    write_json(
        path,
        {
            "schema_version": 1,
            "environment": {"model": args.model},
            "matrix": spec,
            "profiles": [row],
        },
    )
    assert load_checkpoint(args, spec) == [row]

    args.warmup = 2
    with pytest.raises(RuntimeError, match="different model or matrix"):
        load_checkpoint(args, matrix_spec(args))


def test_no_resume_ignores_existing_checkpoint(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("not json")
    args = _profile_args(path)
    args.no_resume = True
    assert load_checkpoint(args, matrix_spec(args)) == []


def test_layer_profiler_selects_first_layer_of_each_hybrid_type():
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert resolve_layer_ids(layer_types, None) == [0, 3]
    assert resolve_layer_ids(layer_types, [3, 0, 3]) == [3, 0]
    with pytest.raises(ValueError, match="out of range"):
        resolve_layer_ids(layer_types, [4])


def test_layer_profile_counts_only_positive_duration_cuda_events():
    profile = FakeProfile(
        events=[
            FakeEvent(
                "elementwise_kernel",
                device_type="DeviceType.CUDA",
                device_us=100.0,
                count=1,
            ),
            FakeEvent(
                "memory_event",
                device_type="DeviceType.CUDA",
                device_us=0.0,
                count=9,
            ),
        ],
        averages=[
            FakeEvent(
                "qwen35_deltanet_recurrence",
                device_us=100.0,
                cpu_us=200.0,
                self_cpu_us=20.0,
                count=1,
            )
        ],
    )
    summary = summarize_layer_profile(profile, repeat=1, wall_time_s=0.001)
    assert summary["kernel_count"] == 1
    assert summary["cuda_time_ms"] == pytest.approx(0.1)
    assert summary["kernel_families"]["elementwise"]["calls"] == 1
    assert summary["range_attribution"]["qwen35_deltanet_recurrence"][
        "cuda_total_ms"
    ] == pytest.approx(0.1)
