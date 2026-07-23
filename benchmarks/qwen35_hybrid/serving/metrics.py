from __future__ import annotations

import math
from statistics import mean, median
from typing import Iterable

from .request import ServingRequest


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_summary(
    values: Iterable[float],
    *,
    include_p99: bool = False,
) -> dict[str, float | None]:
    clean = [float(value) for value in values]
    result = {
        "mean": mean(clean) if clean else None,
        "p50": median(clean) if clean else None,
        "p95": percentile(clean, 0.95),
    }
    if include_p99:
        result["p99"] = percentile(clean, 0.99)
    return result


class ServingMetrics:
    def __init__(self, requests: list[ServingRequest]):
        self.requests = {request.request_id: request for request in requests}
        self.engine_steps = 0
        self.max_waiting_requests = 0
        self.max_running_requests = 0
        self.max_pending_requests = 0
        self.max_inflight_requests = 0

    @property
    def completed_requests(self) -> int:
        return sum(request.is_finished for request in self.requests.values())

    @property
    def failed_requests(self) -> int:
        return sum(request.is_failed for request in self.requests.values())

    @property
    def rejected_requests(self) -> int:
        return sum(request.is_rejected for request in self.requests.values())

    def record_token(
        self,
        request_id: int,
        token_id: int,
        timestamp: float,
        finished: bool,
    ) -> None:
        self.requests[request_id].record_token(token_id, timestamp, finished)

    def record_prefill_start(
        self,
        request_id: int,
        timestamp: float,
    ) -> None:
        self.requests[request_id].record_prefill_start(timestamp)

    def record_failure(
        self,
        request_id: int,
        timestamp: float,
        reason: str,
    ) -> None:
        self.requests[request_id].mark_failed(timestamp, reason)

    def record_rejection(
        self,
        request_id: int,
        timestamp: float,
        reason: str,
    ) -> None:
        self.requests[request_id].mark_rejected(timestamp, reason)

    def fail_submitted_requests(self, timestamp: float, reason: str) -> None:
        for request in self.requests.values():
            if (
                request.submitted_time is not None
                and not request.is_terminal
            ):
                request.mark_failed(timestamp, reason)

    def observe_engine(
        self,
        waiting: int,
        running: int,
        *,
        step_completed: bool = True,
    ) -> None:
        if step_completed:
            self.engine_steps += 1
        self.max_waiting_requests = max(self.max_waiting_requests, waiting)
        self.max_running_requests = max(self.max_running_requests, running)

    def observe_admission(self, pending: int, inflight: int) -> None:
        self.max_pending_requests = max(self.max_pending_requests, pending)
        self.max_inflight_requests = max(self.max_inflight_requests, inflight)

    @staticmethod
    def _latencies(
        requests: list[ServingRequest],
    ) -> dict[str, dict[str, float | None]]:
        ttft_ms = [
            request.ttft_s * 1000
            for request in requests
            if request.ttft_s is not None
        ]
        itl_ms = [
            interval * 1000
            for request in requests
            for interval in request.itl_s
        ]
        e2e_ms = [
            request.e2e_latency_s * 1000
            for request in requests
            if request.e2e_latency_s is not None
        ]
        queue_ms = [
            request.queue_delay_s * 1000
            for request in requests
            if request.queue_delay_s is not None
        ]
        admission_ms = [
            request.admission_delay_s * 1000
            for request in requests
            if request.admission_delay_s is not None
        ]
        prefill_ms = [
            request.prefill_time_s * 1000
            for request in requests
            if request.prefill_time_s is not None
        ]
        decode_ms = [
            request.decode_time_s * 1000
            for request in requests
            if request.decode_time_s is not None
        ]
        return {
            "ttft_ms": latency_summary(ttft_ms, include_p99=True),
            # TTFT ends at token 1. ITL starts with token 2 - token 1, so the
            # arrival/prefill-to-first-token interval is never mixed into ITL.
            "itl_ms": latency_summary(itl_ms, include_p99=True),
            "e2e_latency_ms": latency_summary(e2e_ms, include_p99=True),
            "admission_delay_ms": latency_summary(
                admission_ms, include_p99=True
            ),
            "queue_time_ms": latency_summary(queue_ms, include_p99=True),
            "prefill_time_ms": latency_summary(
                prefill_ms, include_p99=True
            ),
            "decode_time_ms": latency_summary(
                decode_ms, include_p99=True
            ),
        }

    def summarize(
        self,
        *,
        wall_time_s: float,
        arrival_duration_s: float,
        peak_memory_gb: float | None,
    ) -> dict[str, object]:
        requests = list(self.requests.values())
        completed = [request for request in requests if request.is_finished]
        failed = [request for request in requests if request.is_failed]
        rejected = [request for request in requests if request.is_rejected]
        incomplete = [
            request for request in requests if not request.is_terminal
        ]
        admitted = [
            request
            for request in requests
            if request.engine_request_id is not None
        ]
        input_tokens = sum(len(request.input_ids) for request in completed)
        output_tokens = sum(len(request.output_tokens) for request in completed)
        total_tokens = input_tokens + output_tokens
        workload_kinds = sorted({request.workload for request in requests})
        by_workload = {}
        for kind in workload_kinds:
            subset = [request for request in completed if request.workload == kind]
            by_workload[kind] = {
                "request_count": len(subset),
                "input_tokens": sum(
                    len(request.input_ids) for request in subset
                ),
                "output_tokens": sum(
                    len(request.output_tokens) for request in subset
                ),
                **self._latencies(subset),
            }
        return {
            "num_requests": len(requests),
            "request_count": len(requests),
            "completed_requests": len(completed),
            "failed_requests": len(failed),
            "rejected_requests": len(rejected),
            "incomplete_requests": len(incomplete),
            "admitted_requests": len(admitted),
            "acceptance_rate": (
                len(admitted) / len(requests) if requests else None
            ),
            "rejection_rate": (
                len(rejected) / len(requests) if requests else None
            ),
            "offered_request_rate_per_s": (
                len(requests) / arrival_duration_s
                if arrival_duration_s > 0
                else None
            ),
            "input_tokens_total": input_tokens,
            "output_tokens_total": output_tokens,
            "total_tokens": total_tokens,
            # Kept for compatibility with the first online benchmark artifact.
            "generated_tokens": output_tokens,
            "arrival_duration_s": arrival_duration_s,
            "wall_time_s": wall_time_s,
            "drain_time_s": max(0.0, wall_time_s - arrival_duration_s),
            "saturated": (
                bool(rejected)
                or bool(failed)
                or max(0.0, wall_time_s - arrival_duration_s)
                > max(1.0, arrival_duration_s * 0.1)
            ),
            "input_tokens_per_s": (
                input_tokens / wall_time_s if wall_time_s > 0 else None
            ),
            "output_tokens_per_s": (
                output_tokens / wall_time_s if wall_time_s > 0 else None
            ),
            "total_tokens_per_s": (
                total_tokens / wall_time_s if wall_time_s > 0 else None
            ),
            "throughput_tokens_per_s": (
                output_tokens / wall_time_s if wall_time_s > 0 else None
            ),
            "request_throughput_per_s": (
                len(completed) / wall_time_s if wall_time_s > 0 else None
            ),
            "peak_memory_gb": peak_memory_gb,
            "engine_steps": self.engine_steps,
            "max_waiting_requests": self.max_waiting_requests,
            "max_running_requests": self.max_running_requests,
            "max_pending_requests": self.max_pending_requests,
            "max_inflight_requests": self.max_inflight_requests,
            **self._latencies(completed),
            "by_workload": by_workload,
            "failed_request_ids": [
                request.request_id for request in failed
            ],
            "rejected_request_ids": [
                request.request_id for request in rejected
            ],
            "incomplete_request_ids": [
                request.request_id for request in incomplete
            ],
        }

    def request_records(self) -> list[dict[str, object]]:
        return [
            self.requests[request_id].to_dict()
            for request_id in sorted(self.requests)
        ]
