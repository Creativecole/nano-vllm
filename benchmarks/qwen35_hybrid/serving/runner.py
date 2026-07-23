from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from .metrics import ServingMetrics
from .request import ServingRequest, TokenEvent


class NanoVLLMAdapter:
    """Translate the synchronous nano-vLLM engine into per-step token events."""

    def __init__(self, llm):
        self.llm = llm
        self._request_ids: dict[int, int] = {}
        self._emitted_tokens: dict[int, int] = {}
        self._prefill_starts_emitted: set[int] = set()
        self.last_prefill_starts: dict[int, float] = {}

    def submit(self, request: ServingRequest) -> int:
        from nanovllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=request.max_output_tokens,
            ignore_eos=True,
        )
        engine_request_id = self.llm.add_request(
            request.input_ids, sampling_params
        )
        request.engine_request_id = engine_request_id
        self._request_ids[engine_request_id] = request.request_id
        self._emitted_tokens[engine_request_id] = 0
        return engine_request_id

    def has_work(self) -> bool:
        return not self.llm.is_finished()

    def inflight_count(self) -> int:
        return len(self._request_ids)

    def step(self) -> list[TokenEvent]:
        tracked_sequences = (
            list(self.llm.scheduler.waiting)
            + list(self.llm.scheduler.running)
        )
        finished_outputs, _ = self.llm.step()
        self.last_prefill_starts = {}
        for seq in tracked_sequences:
            started_at = getattr(seq, "prefill_started_at", None)
            if (
                started_at is not None
                and seq.seq_id not in self._prefill_starts_emitted
                and seq.seq_id in self._request_ids
            ):
                self.last_prefill_starts[
                    self._request_ids[seq.seq_id]
                ] = started_at
                self._prefill_starts_emitted.add(seq.seq_id)
        finished = {seq_id: tokens for seq_id, tokens in finished_outputs}
        active = {
            seq.seq_id: list(seq.completion_token_ids)
            for seq in (
                list(self.llm.scheduler.waiting)
                + list(self.llm.scheduler.running)
            )
        }
        events = []
        for engine_request_id in sorted(set(active) | set(finished)):
            tokens = (
                finished[engine_request_id]
                if engine_request_id in finished
                else active[engine_request_id]
            )
            emitted = self._emitted_tokens[engine_request_id]
            if len(tokens) < emitted:
                raise RuntimeError(
                    f"Completion token count moved backwards for request "
                    f"{engine_request_id}: {len(tokens)} < {emitted}"
                )
            new_tokens = tokens[emitted:]
            is_finished = engine_request_id in finished
            for index, token_id in enumerate(new_tokens):
                events.append(
                    TokenEvent(
                        request_id=self._request_ids[engine_request_id],
                        token_id=token_id,
                        finished=is_finished and index == len(new_tokens) - 1,
                    )
                )
            self._emitted_tokens[engine_request_id] = len(tokens)
            if is_finished:
                if not new_tokens:
                    raise RuntimeError(
                        f"Finished request {engine_request_id} emitted no final token"
                    )
                del self._request_ids[engine_request_id]
                del self._emitted_tokens[engine_request_id]
                self._prefill_starts_emitted.discard(engine_request_id)
        return events

    def queue_depths(self) -> tuple[int, int]:
        return (
            len(self.llm.scheduler.waiting),
            len(self.llm.scheduler.running),
        )


def run_online_workload(
    *,
    adapter,
    requests: list[ServingRequest],
    metrics: ServingMetrics,
    clock: Callable[[], float] = time.perf_counter,
    sleeper: Callable[[float], None] = time.sleep,
    progress: Callable[[str], None] | None = None,
    arrival_window_s: float | None = None,
    max_queue_size: int = 32,
    max_inflight_requests: int = 8,
) -> float:
    """Run bounded open-loop admission and drain all accepted work."""
    requests = sorted(requests, key=lambda request: request.arrival_time)
    if arrival_window_s is None:
        arrival_window_s = requests[-1].arrival_time if requests else 0.0
    if arrival_window_s < 0:
        raise ValueError("arrival_window_s must be non-negative")
    if max_queue_size < 0:
        raise ValueError("max_queue_size must be non-negative")
    if max_inflight_requests <= 0:
        raise ValueError("max_inflight_requests must be positive")
    start = clock()
    next_request = 0
    admitted = 0
    pending: deque[ServingRequest] = deque()

    def submit(request: ServingRequest, timestamp: float) -> None:
        nonlocal admitted
        request.submitted_time = timestamp
        try:
            adapter.submit(request)
        except Exception as exc:
            metrics.record_failure(
                request.request_id,
                timestamp,
                f"{type(exc).__name__}: {exc}",
            )
            if progress is not None:
                progress(
                    f"request={request.request_id} failed during submit: "
                    f"{type(exc).__name__}: {exc}"
                )
        else:
            admitted += 1

    while True:
        now = clock() - start
        arrivals = 0
        while (
            next_request < len(requests)
            and requests[next_request].arrival_time <= now
        ):
            request = requests[next_request]
            if (
                not pending
                and adapter.inflight_count() < max_inflight_requests
            ):
                submit(request, now)
            elif len(pending) < max_queue_size:
                pending.append(request)
            else:
                metrics.record_rejection(
                    request.request_id,
                    now,
                    f"admission queue full (capacity={max_queue_size})",
                )
            next_request += 1
            arrivals += 1

        while (
            pending
            and adapter.inflight_count() < max_inflight_requests
        ):
            submit(pending.popleft(), now)

        metrics.observe_admission(
            len(pending),
            adapter.inflight_count(),
        )
        if progress is not None and arrivals:
            progress(
                f"offered={next_request}/{len(requests)} "
                f"admitted={admitted} "
                f"rejected={metrics.rejected_requests} "
                f"pending={len(pending)} "
                f"inflight={adapter.inflight_count()} "
                f"completed={metrics.completed_requests}"
            )

        if adapter.has_work():
            waiting, running = adapter.queue_depths()
            metrics.observe_engine(
                waiting,
                running,
                step_completed=False,
            )
            events = adapter.step()
            event_time = clock() - start
            for request_id, started_at in getattr(
                adapter, "last_prefill_starts", {}
            ).items():
                metrics.record_prefill_start(
                    request_id,
                    max(0.0, started_at - start),
                )
            for event in events:
                metrics.record_token(
                    event.request_id,
                    event.token_id,
                    event_time,
                    event.finished,
                )
            waiting, running = adapter.queue_depths()
            metrics.observe_engine(waiting, running)
            continue

        if pending:
            raise RuntimeError(
                "Admission queue is non-empty while the engine has no work"
            )

        if next_request < len(requests):
            delay = requests[next_request].arrival_time - (clock() - start)
            if delay > 0:
                sleeper(delay)
            continue

        if clock() - start < arrival_window_s:
            sleeper(arrival_window_s - (clock() - start))
            continue

        if metrics.completed_requests < admitted:
            raise RuntimeError("Engine lost admitted requests before completion")

        break

    return clock() - start
