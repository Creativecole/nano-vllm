#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import DEFAULT_RESULTS_DIR  # noqa: E402
from benchmarks.qwen35_hybrid.serving.launch import (  # noqa: E402
    build_serving_case,
    run_serving_case,
)


def parse_rates(value: str) -> list[float]:
    rates = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("rates must be comma-separated positive numbers")
    return rates


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an open-loop request-rate sweep in isolated processes."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--backend",
        choices=("nanovllm_sequential", "nanovllm_chunked"),
        default="nanovllm_chunked",
    )
    parser.add_argument(
        "--workload",
        choices=("chat", "long_context", "mixed"),
        default="mixed",
    )
    parser.add_argument("--request-rates", default="0.5,1,2,4,8")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-inflight-requests", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument(
        "--scheduler-policy",
        choices=("prefill_first", "interleave", "unified"),
        default="prefill_first",
    )
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT / "docs/qwen35_hybrid/serving_runs",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def row_from_result(result) -> dict[str, object]:
    if result.summary_payload is None:
        return {
            "rate": result.case.request_rate,
            "backend": result.case.backend,
            "status": "dry-run" if result.returncode == 0 else "failed",
            "error": result.error,
        }
    summary = result.summary_payload["summary"]
    return {
        "rate": result.case.request_rate,
        "backend": result.case.backend,
        "scheduler_policy": result.summary_payload["configuration"][
            "scheduler_policy"
        ],
        "max_prefill_chunk_tokens": result.summary_payload[
            "configuration"
        ]["max_prefill_chunk_tokens"],
        "offered_requests": summary["num_requests"],
        "admitted_requests": summary["admitted_requests"],
        "completed_requests": summary["completed_requests"],
        "failed_requests": summary["failed_requests"],
        "rejected_requests": summary["rejected_requests"],
        "acceptance_rate": summary["acceptance_rate"],
        "output_tok_s": summary["output_tokens_per_s"],
        "request_throughput_per_s": summary["request_throughput_per_s"],
        "p50_ttft_ms": summary["ttft_ms"]["p50"],
        "p95_ttft_ms": summary["ttft_ms"]["p95"],
        "p95_itl_ms": summary["itl_ms"]["p95"],
        "max_pending_queue": summary["max_pending_requests"],
        "max_inflight": summary["max_inflight_requests"],
        "drain_time_s": summary["drain_time_s"],
        "saturated": summary["saturated"],
        "status": "ok" if result.returncode == 0 else "failed",
        "error": result.error,
        "raw_trace": str(result.case.raw_json),
        "summary_json": str(result.case.summary_json),
    }


def main():
    args = parse_args()
    rates = parse_rates(args.request_rates)
    rows = []
    for rate in rates:
        case = build_serving_case(
            model=args.model,
            backend=args.backend,
            workload=args.workload,
            request_rate=rate,
            duration=args.duration,
            seed=args.seed,
            max_num_seqs=args.max_num_seqs,
            max_inflight_requests=args.max_inflight_requests,
            max_queue_size=args.max_queue_size,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            deltanet_chunk_size=args.deltanet_chunk_size,
            scheduler_policy=args.scheduler_policy,
            max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
            warmup_requests=args.warmup_requests,
            output_dir=args.output_dir,
            summary_dir=args.summary_dir,
        )
        rows.append(row_from_result(run_serving_case(case, dry_run=args.dry_run)))

    output_csv = args.output_csv
    if output_csv is None:
        suffix = ""
        if args.scheduler_policy in ("interleave", "unified"):
            suffix = (
                f"_{args.scheduler_policy}_c"
                f"{args.max_prefill_chunk_tokens}"
            )
        output_csv = DEFAULT_RESULTS_DIR / (
            f"online_serving_rate_sweep{suffix}.csv"
        )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rate",
        "backend",
        "scheduler_policy",
        "max_prefill_chunk_tokens",
        "offered_requests",
        "admitted_requests",
        "completed_requests",
        "failed_requests",
        "rejected_requests",
        "acceptance_rate",
        "output_tok_s",
        "request_throughput_per_s",
        "p50_ttft_ms",
        "p95_ttft_ms",
        "p95_itl_ms",
        "max_pending_queue",
        "max_inflight",
        "drain_time_s",
        "saturated",
        "status",
        "error",
        "raw_trace",
        "summary_json",
    ]
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {output_csv}", flush=True)


if __name__ == "__main__":
    main()
