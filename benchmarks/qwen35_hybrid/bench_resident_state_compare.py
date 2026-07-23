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


def parse_args():
    parser = argparse.ArgumentParser(
        description="A/B compare resident DeltaNet state against gather/commit."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--workload", default="mixed")
    parser.add_argument("--request-rate", type=float, default=2.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-inflight-requests", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RESULTS_DIR / "resident_state_ab"),
    )
    parser.add_argument(
        "--summary-dir",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/resident_state_ab"),
    )
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "resident_state_ab.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/resident_state_ab.md"),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_case(args, resident_deltanet_state: bool):
    return build_serving_case(
        model=args.model,
        backend="nanovllm_chunked",
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
        scheduler_policy="prefill_first",
        max_prefill_chunk_tokens=256,
        warmup_requests=args.warmup_requests,
        output_dir=Path(args.output_dir),
        summary_dir=Path(args.summary_dir),
        resident_deltanet_state=resident_deltanet_state,
    )


def metric(summary: dict, section: str, field: str):
    return (summary.get(section) or {}).get(field)


def result_row(result, resident: bool):
    payload = result.summary_payload or {}
    summary = payload.get("summary") or {}
    cache = payload.get("cache_stats") or {}
    execution = cache.get("execution") or {}
    state_execution = cache.get("state_execution") or {}
    return {
        "state_path": "resident" if resident else "gather_commit",
        "completed_requests": summary.get("completed_requests"),
        "rejected_requests": summary.get("rejected_requests"),
        "ttft_p50_ms": metric(summary, "ttft_ms", "p50"),
        "ttft_p95_ms": metric(summary, "ttft_ms", "p95"),
        "itl_p95_ms": metric(summary, "itl_ms", "p95"),
        "output_tokens_per_s": summary.get("output_tokens_per_s"),
        "peak_memory_gb": summary.get("peak_memory_gb"),
        "state_gather_calls": execution.get("state_gather_calls"),
        "state_commit_calls": execution.get("state_commit_calls"),
        "resident_view_calls": execution.get("state_resident_view_calls"),
        "allocation_zero_ops": state_execution.get("allocation_zero_ops"),
        "estimated_state_copy_bytes_avoided": state_execution.get(
            "estimated_state_copy_bytes_avoided"
        ),
        "compaction_bytes": state_execution.get("compaction_bytes"),
        "error": result.error,
        "summary_json": str(result.case.summary_json),
    }


def render_markdown(payload):
    rows = payload["results"]
    table = markdown_table(
        [
            "State path",
            "Completed",
            "Rejected",
            "TTFT p50 ms",
            "TTFT p95 ms",
            "ITL p95 ms",
            "Output tok/s",
            "Gather calls",
            "Commit calls",
            "Resident views",
            "Allocation zero ops",
            "Avoided state-copy bytes",
            "Compaction bytes",
        ],
        [
            [
                row["state_path"],
                row["completed_requests"],
                row["rejected_requests"],
                row["ttft_p50_ms"],
                row["ttft_p95_ms"],
                row["itl_p95_ms"],
                row["output_tokens_per_s"],
                row["state_gather_calls"],
                row["state_commit_calls"],
                row["resident_view_calls"],
                row["allocation_zero_ops"],
                row["estimated_state_copy_bytes_avoided"],
                row["compaction_bytes"],
            ]
            for row in rows
        ],
    )
    return f"""# Qwen3.5 Resident DeltaNet State A/B

The two subprocesses use the same model, workload, Poisson arrival seed, request
rate, queue limits, and `prefill_first` scheduler. The only runtime difference is
whether DeltaNet state is read through resident pool views or materialized with
`index_select` and committed with `index_copy_`.

{table}

`Avoided state-copy bytes` is an estimate of the logical gather plus commit
traffic removed by resident views. Request-completion compaction is reported
separately.
"""


def main():
    args = parse_args()
    cases = [
        (False, build_case(args, False)),
        (True, build_case(args, True)),
    ]
    rows = []
    for resident, case in cases:
        result = run_serving_case(case, dry_run=args.dry_run)
        if not args.dry_run:
            rows.append(result_row(result, resident))
    if args.dry_run:
        return
    payload = {
        "configuration": {
            "model": args.model,
            "workload": args.workload,
            "request_rate": args.request_rate,
            "duration_s": args.duration,
            "seed": args.seed,
            "scheduler_policy": "prefill_first",
            "deltanet_backend": "chunked",
            "deltanet_chunk_size": args.deltanet_chunk_size,
        },
        "results": rows,
    }
    write_json(args.save_json, payload)
    write_text(args.save_md, render_markdown(payload))
    print(f"Saved {args.save_json}", flush=True)
    print(f"Saved {args.save_md}", flush=True)


if __name__ == "__main__":
    main()
