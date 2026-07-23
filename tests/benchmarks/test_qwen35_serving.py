from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from benchmarks.qwen35_hybrid.serving.generator import (
    generate_open_loop_requests,
    workload_metadata,
)
from benchmarks.qwen35_hybrid.serving.metrics import ServingMetrics
from benchmarks.qwen35_hybrid.serving.request import ServingRequest, TokenEvent
from benchmarks.qwen35_hybrid.serving.runner import (
    NanoVLLMAdapter,
    run_online_workload,
)


def test_open_loop_generator_is_seeded_and_respects_workload_shapes():
    first = generate_open_loop_requests(
        request_rate=4,
        duration_s=2,
        workload="mixed",
        vocab_size=256,
        seed=11,
        max_requests=8,
    )
    second = generate_open_loop_requests(
        request_rate=4,
        duration_s=2,
        workload="mixed",
        vocab_size=256,
        seed=11,
        max_requests=8,
    )

    assert [
        (request.arrival_time, request.workload, request.input_ids)
        for request in first
    ] == [
        (request.arrival_time, request.workload, request.input_ids)
        for request in second
    ]
    assert first[0].arrival_time == 0
    assert all(
        request.arrival_time <= next_request.arrival_time
        for request, next_request in zip(first, first[1:])
    )
    for request in first:
        if request.workload == "chat":
            assert 128 <= len(request.input_ids) <= 512
            assert request.max_output_tokens == 128
        else:
            assert request.workload == "long_context"
            assert 2048 <= len(request.input_ids) <= 8192
            assert request.max_output_tokens == 64


def test_request_metrics_use_arrival_and_adjacent_token_times():
    request = ServingRequest(0, [1, 2], 1.0, 3, "chat")
    request.submitted_time = 1.2
    request.record_prefill_start(1.3)
    request.record_token(10, 1.5, False)
    request.record_token(11, 1.7, False)
    request.record_token(12, 2.0, True)

    assert request.ttft_s == pytest.approx(0.5)
    assert request.itl_s == pytest.approx([0.2, 0.3])
    assert request.e2e_latency_s == pytest.approx(1.0)
    assert request.admission_delay_s == pytest.approx(0.2)
    assert request.queue_delay_s == pytest.approx(0.1)
    assert request.prefill_time_s == pytest.approx(0.2)
    assert request.decode_time_s == pytest.approx(0.5)

    metrics = ServingMetrics([request])
    summary = metrics.summarize(
        wall_time_s=2.0,
        arrival_duration_s=1.0,
        peak_memory_gb=3.0,
    )
    assert summary["throughput_tokens_per_s"] == 1.5
    assert summary["input_tokens_total"] == 2
    assert summary["output_tokens_total"] == 3
    assert summary["input_tokens_per_s"] == 1.0
    assert summary["output_tokens_per_s"] == 1.5
    assert summary["total_tokens_per_s"] == 2.5
    assert summary["failed_requests"] == 0
    assert summary["ttft_ms"]["p95"] == pytest.approx(500)
    assert summary["itl_ms"]["mean"] == pytest.approx(250)
    assert summary["itl_ms"]["p99"] == pytest.approx(299)
    assert summary["e2e_latency_ms"]["p95"] == pytest.approx(1000)
    assert summary["admission_delay_ms"]["p95"] == pytest.approx(200)
    assert summary["queue_time_ms"]["p95"] == pytest.approx(100)
    assert summary["prefill_time_ms"]["p95"] == pytest.approx(200)
    assert summary["decode_time_ms"]["p95"] == pytest.approx(500)


def test_workload_metadata_records_observed_shapes():
    requests = [
        ServingRequest(0, [1] * 128, 0.0, 128, "chat"),
        ServingRequest(1, [1] * 512, 0.5, 128, "chat"),
        ServingRequest(2, [1] * 2048, 1.0, 64, "long_context"),
    ]
    metadata = workload_metadata(
        requests,
        workload="mixed",
        request_rate=2,
        duration_s=60,
        seed=17,
    )

    assert metadata["arrival_process"] == "seeded Poisson open-loop"
    assert metadata["observed_distribution"]["chat"]["request_count"] == 2
    assert metadata["observed_distribution"]["chat"]["prompt_length"] == {
        "count": 2,
        "min": 128,
        "max": 512,
        "mean": 320,
    }
    assert (
        metadata["observed_distribution"]["long_context"]["output_length"][
            "mean"
        ]
        == 64
    )


def test_failed_request_is_not_counted_as_completed_throughput():
    completed = ServingRequest(0, [1, 2], 0.0, 1, "chat")
    completed.record_token(10, 0.5, True)
    failed = ServingRequest(1, [1, 2, 3], 0.1, 1, "chat")
    failed.submitted_time = 0.2
    failed.mark_failed(0.3, "synthetic failure")

    summary = ServingMetrics([completed, failed]).summarize(
        wall_time_s=1.0,
        arrival_duration_s=1.0,
        peak_memory_gb=None,
    )
    assert summary["num_requests"] == 2
    assert summary["completed_requests"] == 1
    assert summary["failed_requests"] == 1
    assert summary["input_tokens_total"] == 2
    assert summary["output_tokens_total"] == 1
    assert summary["failed_request_ids"] == [1]


def test_bounded_admission_rejects_when_queue_is_full():
    clock = FakeClock()
    requests = [
        ServingRequest(request_id, [request_id + 1], 0.0, 1, "chat")
        for request_id in range(3)
    ]
    metrics = ServingMetrics(requests)
    elapsed = run_online_workload(
        adapter=FakeAdapter(clock),
        requests=requests,
        metrics=metrics,
        clock=clock,
        sleeper=clock.advance,
        max_inflight_requests=1,
        max_queue_size=1,
    )
    summary = metrics.summarize(
        wall_time_s=elapsed,
        arrival_duration_s=0.1,
        peak_memory_gb=None,
    )

    assert summary["completed_requests"] == 2
    assert summary["rejected_requests"] == 1
    assert summary["failed_requests"] == 0
    assert summary["acceptance_rate"] == pytest.approx(2 / 3)
    assert summary["max_pending_requests"] == 1
    assert summary["max_inflight_requests"] == 1
    assert requests[2].to_dict()["status"] == "rejected"


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, duration):
        self.value += duration


class FakeAdapter:
    def __init__(self, clock):
        self.clock = clock
        self.active = {}

    def submit(self, request):
        request.engine_request_id = request.request_id
        self.active[request.request_id] = {
            "remaining": request.max_output_tokens,
            "next_token": 100,
        }
        return request.request_id

    def has_work(self):
        return bool(self.active)

    def inflight_count(self):
        return len(self.active)

    def step(self):
        self.clock.advance(0.1)
        events = []
        for request_id in list(self.active):
            state = self.active[request_id]
            state["remaining"] -= 1
            finished = state["remaining"] == 0
            events.append(
                TokenEvent(request_id, state["next_token"], finished)
            )
            state["next_token"] += 1
            if finished:
                del self.active[request_id]
        return events

    def queue_depths(self):
        return 0, len(self.active)


def test_online_loop_submits_dynamic_arrivals_and_drains():
    clock = FakeClock()
    requests = [
        ServingRequest(0, [1], 0.0, 2, "chat"),
        ServingRequest(1, [2], 0.15, 1, "long_context"),
    ]
    metrics = ServingMetrics(requests)
    elapsed = run_online_workload(
        adapter=FakeAdapter(clock),
        requests=requests,
        metrics=metrics,
        clock=clock,
        sleeper=clock.advance,
    )

    assert elapsed == pytest.approx(0.3)
    assert metrics.completed_requests == 2
    assert requests[0].token_times == pytest.approx([0.1, 0.2])
    # Request 1 arrives while the second synchronous step is running and is
    # submitted at t=0.2, so its queue delay contributes to TTFT.
    assert requests[1].submitted_time == pytest.approx(0.2)
    assert requests[1].ttft_s == pytest.approx(0.15)


def test_nanovllm_adapter_emits_active_and_finished_token_deltas():
    sequence = SimpleNamespace(seq_id=7, completion_token_ids=[101])
    scheduler = SimpleNamespace(waiting=deque(), running=deque([sequence]))
    llm = SimpleNamespace(scheduler=scheduler)
    adapter = NanoVLLMAdapter(llm)
    adapter._request_ids[7] = 3
    adapter._emitted_tokens[7] = 0

    llm.step = lambda: ([], -1)
    assert adapter.step() == [TokenEvent(3, 101, False)]

    scheduler.running.clear()
    llm.step = lambda: ([(7, [101, 102])], -1)
    assert adapter.step() == [TokenEvent(3, 102, True)]
    assert not adapter._request_ids
    assert not adapter._emitted_tokens


def test_generator_rejects_invalid_rate_and_workload():
    with pytest.raises(ValueError, match="request_rate"):
        generate_open_loop_requests(
            request_rate=0,
            duration_s=1,
            workload="chat",
            vocab_size=32,
            seed=1,
        )
    with pytest.raises(ValueError, match="Unsupported workload"):
        generate_open_loop_requests(
            request_rate=1,
            duration_s=1,
            workload="batch",
            vocab_size=32,
            seed=1,
        )
