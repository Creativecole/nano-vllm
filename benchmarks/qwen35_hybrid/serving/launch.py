from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_SERVING = REPO_ROOT / "benchmarks/qwen35_hybrid/bench_serving.py"


def rate_slug(rate: float) -> str:
    return str(rate).replace(".", "p")


@dataclass(frozen=True)
class ServingCase:
    backend: str
    request_rate: float
    raw_json: Path
    summary_json: Path
    summary_md: Path
    command: list[str]


@dataclass(frozen=True)
class ServingCaseResult:
    case: ServingCase
    returncode: int
    summary_payload: dict[str, object] | None
    error: str | None


def build_serving_case(
    *,
    model: str,
    backend: str,
    workload: str,
    request_rate: float,
    duration: float,
    seed: int,
    max_num_seqs: int,
    max_inflight_requests: int,
    max_queue_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    deltanet_chunk_size: int,
    scheduler_policy: str,
    max_prefill_chunk_tokens: int,
    warmup_requests: int,
    output_dir: Path,
    summary_dir: Path,
    max_partial_prefills: int = 1,
    max_long_partial_prefills: int = 1,
    long_prefill_token_threshold: int = 0,
    decode_reserve_blocks_per_seq: int = 1,
    resident_deltanet_state: bool = True,
) -> ServingCase:
    output_dir = output_dir.resolve()
    summary_dir = summary_dir.resolve()
    policy = scheduler_policy
    if scheduler_policy in ("interleave", "unified"):
        policy = f"{policy}_c{max_prefill_chunk_tokens}"
    state_path = "resident" if resident_deltanet_state else "statecopy"
    stem = (
        f"online_{backend}_{policy}_{state_path}_{workload}_"
        f"r{rate_slug(request_rate)}"
    )
    raw_json = output_dir / f"{stem}.json"
    summary_json = output_dir / f"{stem}_summary.json"
    summary_md = summary_dir / f"{stem}_summary.md"
    command = [
        sys.executable,
        str(BENCH_SERVING),
        "--model",
        model,
        "--backend",
        backend,
        "--workload",
        workload,
        "--request-rate",
        str(request_rate),
        "--duration",
        str(duration),
        "--seed",
        str(seed),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-inflight-requests",
        str(max_inflight_requests),
        "--max-queue-size",
        str(max_queue_size),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--deltanet-chunk-size",
        str(deltanet_chunk_size),
        "--scheduler-policy",
        scheduler_policy,
        "--max-prefill-chunk-tokens",
        str(max_prefill_chunk_tokens),
        "--max-partial-prefills",
        str(max_partial_prefills),
        "--max-long-partial-prefills",
        str(max_long_partial_prefills),
        "--long-prefill-token-threshold",
        str(long_prefill_token_threshold),
        "--decode-reserve-blocks-per-seq",
        str(decode_reserve_blocks_per_seq),
        "--warmup-requests",
        str(warmup_requests),
        "--save-json",
        str(raw_json),
        "--save-summary-json",
        str(summary_json),
        "--save-summary-md",
        str(summary_md),
    ]
    if not resident_deltanet_state:
        command.append("--disable-resident-deltanet-state")
    return ServingCase(
        backend=backend,
        request_rate=request_rate,
        raw_json=raw_json,
        summary_json=summary_json,
        summary_md=summary_md,
        command=command,
    )


def run_serving_case(
    case: ServingCase,
    *,
    dry_run: bool = False,
) -> ServingCaseResult:
    print("[serving-case] " + " ".join(case.command), flush=True)
    if dry_run:
        return ServingCaseResult(case, 0, None, None)
    completed = subprocess.run(case.command, cwd=REPO_ROOT, check=False)
    payload = None
    error = None
    if case.summary_json.exists():
        payload = json.loads(case.summary_json.read_text())
        error = payload.get("benchmark_error")
    elif completed.returncode != 0:
        error = (
            f"benchmark process exited with code {completed.returncode} "
            "without a summary artifact"
        )
    return ServingCaseResult(
        case=case,
        returncode=completed.returncode,
        summary_payload=payload,
        error=error,
    )
