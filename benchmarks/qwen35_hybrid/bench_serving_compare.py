#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    markdown_table,
    write_json,
    write_text,
)
from benchmarks.qwen35_hybrid.serving.launch import (  # noqa: E402
    build_serving_case,
    run_serving_case,
)


BACKENDS = ("nanovllm_sequential", "nanovllm_chunked")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare sequential and chunked online serving backends."
    )
    parser.add_argument("--model", default="../models/Qwen3.5-9B")
    parser.add_argument("--workload", default="mixed", choices=("mixed",))
    parser.add_argument("--request-rate", type=float, default=2.0)
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
        "--save-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "online_serving_backend_compare.json",
    )
    parser.add_argument(
        "--save-md",
        type=Path,
        default=REPO_ROOT
        / "docs/qwen35_hybrid/online_serving_backend_compare.md",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def comparison_row(result) -> dict[str, object]:
    if result.summary_payload is None:
        return {
            "backend": result.case.backend,
            "status": "dry-run" if result.returncode == 0 else "failed",
            "error": result.error,
        }
    summary = result.summary_payload["summary"]
    return {
        "backend": result.case.backend,
        "output_tokens_per_s": summary["output_tokens_per_s"],
        "ttft_p50_ms": summary["ttft_ms"]["p50"],
        "ttft_p95_ms": summary["ttft_ms"]["p95"],
        "itl_p95_ms": summary["itl_ms"]["p95"],
        "completed_requests": summary["completed_requests"],
        "failed_requests": summary["failed_requests"],
        "rejected_requests": summary["rejected_requests"],
        "acceptance_rate": summary["acceptance_rate"],
        "peak_memory_gb": summary["peak_memory_gb"],
        "status": "ok" if result.returncode == 0 else "failed",
        "error": result.error,
        "raw_trace": str(result.case.raw_json),
        "summary_json": str(result.case.summary_json),
    }


def render_markdown(args, rows: list[dict[str, object]]) -> str:
    table = markdown_table(
        [
            "Backend",
            "Output tok/s",
            "P50 TTFT ms",
            "P95 TTFT ms",
            "P95 ITL ms",
            "Completed",
            "Failed",
            "Rejected",
            "Accept rate",
            "Status",
        ],
        [
            [
                row.get("backend"),
                row.get("output_tokens_per_s"),
                row.get("ttft_p50_ms"),
                row.get("ttft_p95_ms"),
                row.get("itl_p95_ms"),
                row.get("completed_requests"),
                row.get("failed_requests"),
                row.get("rejected_requests"),
                row.get("acceptance_rate"),
                row.get("status"),
            ]
            for row in rows
        ],
    )
    return f"""# Sequential vs Chunked Online Serving

Qwen3.5-9B, BF16, single RTX 5090, `{args.workload}` workload,
{args.request_rate} requests/s, {args.duration}s open-loop arrival window, seed
{args.seed}. Each backend runs in an isolated process with the same generated request
trace.

{table}

Chunked execution changes DeltaNet prefill. Decode remains the single-token recurrent
path, so the primary online signal is TTFT under dynamic arrivals; output throughput
and ITL are retained as workload-level regression checks. Raw traces and per-backend
summary JSON files are linked in the aggregate JSON artifact.
"""


def main():
    args = parse_args()
    rows = []
    for backend in BACKENDS:
        case = build_serving_case(
            model=args.model,
            backend=backend,
            workload=args.workload,
            request_rate=args.request_rate,
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
        rows.append(
            comparison_row(run_serving_case(case, dry_run=args.dry_run))
        )

    payload = {
        "configuration": {
            "model": args.model,
            "gpu": "NVIDIA GeForce RTX 5090",
            "dtype": "torch.bfloat16",
            "workload": args.workload,
            "request_rate": args.request_rate,
            "duration_s": args.duration,
            "seed": args.seed,
            "max_inflight_requests": args.max_inflight_requests,
            "max_queue_size": args.max_queue_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "scheduler_policy": args.scheduler_policy,
            "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
        },
        "rows": rows,
    }
    write_json(args.save_json, payload)
    write_text(args.save_md, render_markdown(args, rows))
    print(f"Saved {args.save_json}", flush=True)
    print(f"Saved {args.save_md}", flush=True)


if __name__ == "__main__":
    main()
