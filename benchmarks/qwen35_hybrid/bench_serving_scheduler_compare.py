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


def parse_positive_ints(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive integers")
    return values


def parse_positive_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError("expected comma-separated positive numbers")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare prefill-first and unified mixed-batch schedules."
    )
    parser.add_argument("--model", default="../models/Qwen3.5-9B")
    parser.add_argument("--request-rates", default="2,3,4")
    parser.add_argument("--prefill-chunk-tokens", default="256")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-inflight-requests", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument("--max-partial-prefills", type=int, default=2)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=1024)
    parser.add_argument("--decode-reserve-blocks-per-seq", type=int, default=1)
    parser.add_argument("--include-naive-interleave", action="store_true")
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )
    parser.add_argument(
        "--summary-dir",
        type=Path,
        default=REPO_ROOT / "docs/qwen35_hybrid/scheduling_runs",
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "prefill_decode_scheduling.json",
    )
    parser.add_argument(
        "--save-md",
        type=Path,
        default=REPO_ROOT
        / "docs/qwen35_hybrid/08_prefill_decode_scheduling.md",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def result_row(result, policy: str, chunk_tokens: int | None):
    label = (
        "current_prefill_first"
        if policy == "prefill_first"
        else f"{policy}_c{chunk_tokens}"
    )
    if result.summary_payload is None:
        return {
            "backend": label,
            "rate": result.case.request_rate,
            "status": "dry-run" if result.returncode == 0 else "failed",
            "error": result.error,
        }
    summary = result.summary_payload["summary"]
    request_count = summary["request_count"]
    return {
        "backend": label,
        "rate": result.case.request_rate,
        "ttft_p50_ms": summary["ttft_ms"]["p50"],
        "ttft_p95_ms": summary["ttft_ms"]["p95"],
        "itl_p95_ms": summary["itl_ms"]["p95"],
        "output_tokens_per_s": summary["output_tokens_per_s"],
        "admission_delay_p95_ms": summary["admission_delay_ms"]["p95"],
        "queue_time_p95_ms": summary["queue_time_ms"]["p95"],
        "prefill_time_p95_ms": summary["prefill_time_ms"]["p95"],
        "completed_requests": summary["completed_requests"],
        "rejected_requests": summary["rejected_requests"],
        "incomplete_requests": summary["incomplete_requests"],
        "completion_rate": (
            summary["completed_requests"] / request_count
            if request_count
            else None
        ),
        "acceptance_rate": summary["acceptance_rate"],
        "saturated": summary["saturated"],
        "status": "ok" if result.returncode == 0 else "failed",
        "error": result.error,
        "raw_trace": str(result.case.raw_json),
        "summary_json": str(result.case.summary_json),
    }


def render_markdown(args, rows):
    table = markdown_table(
        [
            "Backend",
            "Rate",
            "TTFT p50",
            "TTFT p95",
            "ITL p95",
            "Output tok/s",
            "Admission p95",
            "Queue p95",
            "Prefill p95",
            "Rejected",
            "Incomplete",
            "Completion",
            "Saturated",
        ],
        [
            [
                row.get("backend"),
                row.get("rate"),
                row.get("ttft_p50_ms"),
                row.get("ttft_p95_ms"),
                row.get("itl_p95_ms"),
                row.get("output_tokens_per_s"),
                row.get("admission_delay_p95_ms"),
                row.get("queue_time_p95_ms"),
                row.get("prefill_time_p95_ms"),
                row.get("rejected_requests"),
                row.get("incomplete_requests"),
                row.get("completion_rate"),
                row.get("saturated"),
            ]
            for row in rows
        ],
    )
    return f"""# Qwen3.5 Prefill/Decode Scheduling

Qwen3.5-9B, BF16, one RTX 5090, `nanovllm_chunked`, mixed workload,
{args.duration}s open-loop arrival window, seed {args.seed}.

`current_prefill_first` runs the original scheduler. `unified_cN` schedules running
decode requests first, spends the remaining shared token budget on bounded prefill
chunks, and submits both groups through one packed ModelRunner execution.
`interleave_cN`, when explicitly requested, is the retained negative-control path that
uses two ModelRunner executions per engine step.

{table}

Admission p95 measures planned arrival to bounded engine admission. Queue p95 measures
engine admission to first prefill scheduling. Prefill p95 measures first prefill
scheduling to first output token. Raw traces are retained for every row; failed and
rejected or incomplete requests are not filtered from the request accounting. Latency
percentiles apply to completed requests, so completion ratio must be read alongside
tail latency.
"""


def main():
    args = parse_args()
    rates = parse_positive_floats(args.request_rates)
    chunks = parse_positive_ints(args.prefill_chunk_tokens)
    rows = []
    cases = [("prefill_first", None)]
    cases.extend(("unified", chunk) for chunk in chunks)
    if args.include_naive_interleave:
        cases.extend(("interleave", chunk) for chunk in chunks)
    for rate in rates:
        for policy, chunk_tokens in cases:
            effective_chunk = chunk_tokens or chunks[0]
            case = build_serving_case(
                model=args.model,
                backend="nanovllm_chunked",
                workload="mixed",
                request_rate=rate,
                duration=args.duration,
                seed=args.seed,
                max_num_seqs=args.max_num_seqs,
                max_inflight_requests=args.max_inflight_requests,
                max_queue_size=args.max_queue_size,
                max_num_batched_tokens=args.max_num_batched_tokens,
                gpu_memory_utilization=args.gpu_memory_utilization,
                deltanet_chunk_size=args.deltanet_chunk_size,
                scheduler_policy=policy,
                max_prefill_chunk_tokens=effective_chunk,
                warmup_requests=args.warmup_requests,
                output_dir=args.output_dir,
                summary_dir=args.summary_dir,
                max_partial_prefills=args.max_partial_prefills,
                max_long_partial_prefills=args.max_long_partial_prefills,
                long_prefill_token_threshold=args.long_prefill_token_threshold,
                decode_reserve_blocks_per_seq=args.decode_reserve_blocks_per_seq,
            )
            result = run_serving_case(case, dry_run=args.dry_run)
            rows.append(result_row(result, policy, chunk_tokens))

    payload = {
        "configuration": {
            "model": args.model,
            "gpu": "NVIDIA GeForce RTX 5090",
            "dtype": "torch.bfloat16",
            "backend": "nanovllm_chunked",
            "workload": "mixed",
            "request_rates": rates,
            "prefill_chunk_tokens": chunks,
            "duration_s": args.duration,
            "seed": args.seed,
            "max_inflight_requests": args.max_inflight_requests,
            "max_queue_size": args.max_queue_size,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_partial_prefills": args.max_partial_prefills,
            "max_long_partial_prefills": args.max_long_partial_prefills,
            "long_prefill_token_threshold": args.long_prefill_token_threshold,
            "decode_reserve_blocks_per_seq": args.decode_reserve_blocks_per_seq,
        },
        "rows": rows,
    }
    write_json(args.save_json, payload)
    write_text(args.save_md, render_markdown(args, rows))
    print(f"Saved {args.save_json}", flush=True)
    print(f"Saved {args.save_md}", flush=True)


if __name__ == "__main__":
    main()
