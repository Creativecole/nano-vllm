from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ServingRequest:
    request_id: int
    input_ids: list[int]
    arrival_time: float
    max_output_tokens: int
    workload: str
    engine_request_id: int | None = None
    submitted_time: float | None = None
    prefill_start_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    failed_time: float | None = None
    failure_reason: str | None = None
    rejected_time: float | None = None
    rejection_reason: str | None = None
    output_tokens: list[int] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)

    def record_token(self, token_id: int, timestamp: float, finished: bool) -> None:
        if self.is_terminal:
            raise RuntimeError(f"Request {self.request_id} is already terminal")
        if self.first_token_time is None:
            self.first_token_time = timestamp
        self.output_tokens.append(int(token_id))
        self.token_times.append(float(timestamp))
        if finished:
            self.finish_time = timestamp

    def mark_failed(self, timestamp: float, reason: str) -> None:
        if self.is_terminal:
            return
        self.failed_time = float(timestamp)
        self.failure_reason = str(reason)

    def mark_rejected(self, timestamp: float, reason: str) -> None:
        if self.is_terminal:
            return
        self.rejected_time = float(timestamp)
        self.rejection_reason = str(reason)

    def record_prefill_start(self, timestamp: float) -> None:
        if self.prefill_start_time is None:
            self.prefill_start_time = float(timestamp)

    @property
    def ttft_s(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def itl_s(self) -> list[float]:
        return [
            current - previous
            for previous, current in zip(self.token_times, self.token_times[1:])
        ]

    @property
    def e2e_latency_s(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    @property
    def queue_delay_s(self) -> float | None:
        if self.submitted_time is None or self.prefill_start_time is None:
            return None
        return self.prefill_start_time - self.submitted_time

    @property
    def admission_delay_s(self) -> float | None:
        if self.submitted_time is None:
            return None
        return self.submitted_time - self.arrival_time

    @property
    def prefill_time_s(self) -> float | None:
        if self.prefill_start_time is None or self.first_token_time is None:
            return None
        return self.first_token_time - self.prefill_start_time

    @property
    def decode_time_s(self) -> float | None:
        if self.first_token_time is None or self.finish_time is None:
            return None
        return self.finish_time - self.first_token_time

    @property
    def is_finished(self) -> bool:
        return self.finish_time is not None

    @property
    def is_failed(self) -> bool:
        return self.failed_time is not None

    @property
    def is_rejected(self) -> bool:
        return self.rejected_time is not None

    @property
    def is_terminal(self) -> bool:
        return self.is_finished or self.is_failed or self.is_rejected

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "engine_request_id": self.engine_request_id,
            "workload": self.workload,
            "prompt_tokens": len(self.input_ids),
            "requested_output_tokens": self.max_output_tokens,
            "generated_tokens": len(self.output_tokens),
            "arrival_time_s": self.arrival_time,
            "submitted_time_s": self.submitted_time,
            "prefill_start_time_s": self.prefill_start_time,
            "first_token_time_s": self.first_token_time,
            "finish_time_s": self.finish_time,
            "failed_time_s": self.failed_time,
            "rejected_time_s": self.rejected_time,
            "status": (
                "completed"
                if self.is_finished
                else "failed"
                if self.is_failed
                else "rejected"
                if self.is_rejected
                else "incomplete"
            ),
            "failure_reason": self.failure_reason,
            "rejection_reason": self.rejection_reason,
            "queue_delay_ms": (
                self.queue_delay_s * 1000
                if self.queue_delay_s is not None
                else None
            ),
            "queue_time_ms": (
                self.queue_delay_s * 1000
                if self.queue_delay_s is not None
                else None
            ),
            "admission_delay_ms": (
                self.admission_delay_s * 1000
                if self.admission_delay_s is not None
                else None
            ),
            "prefill_time_ms": (
                self.prefill_time_s * 1000
                if self.prefill_time_s is not None
                else None
            ),
            "decode_time_ms": (
                self.decode_time_s * 1000
                if self.decode_time_s is not None
                else None
            ),
            "ttft_ms": self.ttft_s * 1000 if self.ttft_s is not None else None,
            "itl_ms": [value * 1000 for value in self.itl_s],
            "e2e_latency_ms": (
                self.e2e_latency_s * 1000
                if self.e2e_latency_s is not None
                else None
            ),
            "output_tokens": list(self.output_tokens),
            "token_times_s": list(self.token_times),
        }


@dataclass(frozen=True, slots=True)
class TokenEvent:
    request_id: int
    token_id: int
    finished: bool = False
