from profile_e2e import summarize_cuda_kernel_categories, summarize_operator_attribution


class FakeEvent:
    def __init__(self, key, count, self_device_time_total=0.0, self_cpu_time_total=0.0):
        self.key = key
        self.count = count
        self.self_device_time_total = self_device_time_total
        self.self_cpu_time_total = self_cpu_time_total


class FakeProfiler:
    def __init__(self, events):
        self._events = events

    def key_averages(self):
        return self._events


def test_operator_attribution_uses_total_device_time_fields():
    prof = FakeProfiler([
        FakeEvent("aten::mm", 10, self_device_time_total=422_000.0, self_cpu_time_total=1_000.0),
        FakeEvent("void flash::flash_fwd_splitkv_kernel", 5, self_device_time_total=21_000.0),
        FakeEvent("_rms_norm_kernel", 2, self_device_time_total=6_000.0),
        FakeEvent("nano_vllm_engine_step", 1, self_device_time_total=999_000.0),
    ])

    rows = {row["category"]: row for row in summarize_operator_attribution(prof)}

    assert rows["Linear/GEMM"]["cuda_total_ms"] == 422.0
    assert "Attention" not in rows
    assert "Profiler wrapper" not in rows


def test_cuda_kernel_categories_skip_high_level_operator_events():
    prof = FakeProfiler([
        FakeEvent("aten::mm", 10, self_device_time_total=422_000.0, self_cpu_time_total=1_000.0),
        FakeEvent("void cutlass::Kernel2<bf16_gemm>", 10, self_device_time_total=351_000.0),
        FakeEvent("void flash::flash_fwd_splitkv_kernel", 5, self_device_time_total=21_000.0),
        FakeEvent("_rms_norm_kernel", 2, self_device_time_total=6_000.0),
        FakeEvent("nano_vllm_engine_step", 1, self_device_time_total=999_000.0),
    ])

    rows = {row["category"]: row for row in summarize_cuda_kernel_categories(prof)}

    assert rows["Linear/GEMM"]["self_cuda_time_ms"] == 351.0
    assert rows["Attention"]["self_cuda_time_ms"] == 21.0
    assert rows["Normalization"]["self_cuda_time_ms"] == 6.0
    assert "Profiler wrapper" not in rows
