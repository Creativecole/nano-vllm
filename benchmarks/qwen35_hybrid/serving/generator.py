from __future__ import annotations

import random
from statistics import mean

from .request import ServingRequest


WORKLOADS = ("chat", "long_context", "mixed")
WORKLOAD_DEFINITIONS = {
    "chat": {
        "prompt_length": "uniform integer [128, 512]",
        "output_length": 128,
    },
    "long_context": {
        "prompt_length": "uniform integer [2048, 8192]",
        "output_length": 64,
    },
    "mixed": {
        "mix": {"chat": 0.8, "long_context": 0.2},
        "component_shapes": {
            "chat": {
                "prompt_length": "uniform integer [128, 512]",
                "output_length": 128,
            },
            "long_context": {
                "prompt_length": "uniform integer [2048, 8192]",
                "output_length": 64,
            },
        },
    },
}


def _sample_shape(rng: random.Random, workload: str) -> tuple[str, int, int]:
    kind = workload
    if workload == "mixed":
        kind = "chat" if rng.random() < 0.8 else "long_context"
    if kind == "chat":
        return kind, rng.randint(128, 512), 128
    if kind == "long_context":
        return kind, rng.randint(2048, 8192), 64
    raise ValueError(f"Unsupported workload {workload!r}")


def generate_open_loop_requests(
    *,
    request_rate: float,
    duration_s: float,
    workload: str,
    vocab_size: int,
    seed: int,
    max_requests: int | None = None,
) -> list[ServingRequest]:
    """Generate a seeded Poisson open-loop arrival stream.

    The first request arrives at t=0. Subsequent inter-arrival times are sampled
    independently from an exponential distribution with the requested mean rate.
    """
    if request_rate <= 0:
        raise ValueError("request_rate must be positive")
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if workload not in WORKLOADS:
        raise ValueError(f"Unsupported workload {workload!r}")
    if vocab_size <= 1:
        raise ValueError("vocab_size must be greater than one")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive")

    rng = random.Random(seed)
    requests = []
    arrival_time = 0.0
    request_id = 0
    while arrival_time < duration_s:
        if max_requests is not None and len(requests) >= max_requests:
            break
        kind, prompt_len, output_len = _sample_shape(rng, workload)
        input_ids = [
            1 + rng.randrange(vocab_size - 1)
            for _ in range(prompt_len)
        ]
        requests.append(
            ServingRequest(
                request_id=request_id,
                input_ids=input_ids,
                arrival_time=arrival_time,
                max_output_tokens=output_len,
                workload=kind,
            )
        )
        request_id += 1
        arrival_time += rng.expovariate(request_rate)
    return requests


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": mean(values),
    }


def workload_metadata(
    requests: list[ServingRequest],
    *,
    workload: str,
    request_rate: float,
    duration_s: float,
    seed: int,
) -> dict[str, object]:
    by_kind = {}
    for kind in ("chat", "long_context"):
        subset = [request for request in requests if request.workload == kind]
        by_kind[kind] = {
            "request_count": len(subset),
            "prompt_length": _distribution(
                [len(request.input_ids) for request in subset]
            ),
            "output_length": _distribution(
                [request.max_output_tokens for request in subset]
            ),
        }
    return {
        "name": workload,
        "definition": WORKLOAD_DEFINITIONS[workload],
        "arrival_process": "seeded Poisson open-loop",
        "request_rate_per_s": request_rate,
        "configured_duration_s": duration_s,
        "seed": seed,
        "generated_requests": len(requests),
        "first_arrival_s": requests[0].arrival_time if requests else None,
        "last_arrival_s": requests[-1].arrival_time if requests else None,
        "observed_distribution": by_kind,
    }
