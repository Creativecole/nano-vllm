#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    environment_metadata,
    load_model_facts,
    markdown_table,
    peak_memory_gb,
    require_cuda,
    reset_peak_memory,
    write_json,
    write_text,
)
from benchmarks.qwen35_hybrid.serving import (  # noqa: E402
    NanoVLLMAdapter,
    ServingMetrics,
    generate_open_loop_requests,
    run_online_workload,
    workload_metadata,
)


BACKENDS = ("nanovllm_sequential", "nanovllm_chunked")
WORKLOADS = ("chat", "long_context", "mixed")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open-loop online serving benchmark for Qwen3.5 nano-vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", choices=BACKENDS, default="nanovllm_chunked")
    parser.add_argument("--workload", choices=WORKLOADS, default="mixed")
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-inflight-requests", type=int, default=8)
    parser.add_argument("--max-queue-size", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument(
        "--disable-resident-deltanet-state",
        action="store_true",
        help="Use materialized DeltaNet state gather/commit for A/B validation.",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=("prefill_first", "interleave", "unified"),
        default="prefill_first",
    )
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=256)
    parser.add_argument("--max-partial-prefills", type=int, default=1)
    parser.add_argument("--max-long-partial-prefills", type=int, default=1)
    parser.add_argument("--long-prefill-token-threshold", type=int, default=0)
    parser.add_argument("--decode-reserve-blocks-per-seq", type=int, default=1)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--save-json")
    parser.add_argument("--save-summary-json")
    parser.add_argument(
        "--save-md",
        help="Compatibility alias for --save-summary-md.",
    )
    parser.add_argument("--save-summary-md")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def validate_args(args) -> None:
    if args.request_rate <= 0 or args.duration <= 0:
        raise ValueError("request rate and duration must be positive")
    if args.max_requests is not None and args.max_requests <= 0:
        raise ValueError("max requests must be positive")
    if args.max_num_seqs <= 0 or args.max_num_batched_tokens <= 0:
        raise ValueError("engine capacities must be positive")
    if args.max_inflight_requests <= 0:
        raise ValueError("max inflight requests must be positive")
    if args.max_inflight_requests > args.max_num_seqs:
        raise ValueError("max inflight requests cannot exceed max_num_seqs")
    if args.max_queue_size < 0:
        raise ValueError("max queue size must be non-negative")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("gpu memory utilization must be between zero and one")
    if args.deltanet_chunk_size <= 0 or args.warmup_requests < 0:
        raise ValueError("chunk size must be positive and warmup non-negative")
    if args.max_prefill_chunk_tokens <= 0:
        raise ValueError("max prefill chunk tokens must be positive")
    if args.max_partial_prefills <= 0:
        raise ValueError("max partial prefills must be positive")
    if not 0 <= args.max_long_partial_prefills <= args.max_partial_prefills:
        raise ValueError(
            "max long partial prefills must be between zero and max partial prefills"
        )
    if args.long_prefill_token_threshold < 0:
        raise ValueError("long prefill token threshold cannot be negative")
    if args.decode_reserve_blocks_per_seq < 0:
        raise ValueError("decode reserve blocks cannot be negative")
    if args.save_md and args.save_summary_md:
        raise ValueError("use only one of --save-md and --save-summary-md")


def default_json_path(args) -> Path:
    rate = str(args.request_rate).replace(".", "p")
    policy = args.scheduler_policy
    if args.scheduler_policy in ("interleave", "unified"):
        policy = f"{policy}_c{args.max_prefill_chunk_tokens}"
    state_path = (
        "statecopy"
        if args.disable_resident_deltanet_state
        else "resident"
    )
    return DEFAULT_RESULTS_DIR / (
        f"online_{args.backend}_{policy}_{state_path}_{args.workload}_r{rate}.json"
    )


def default_summary_json_path(raw_path: Path) -> Path:
    return raw_path.with_name(f"{raw_path.stem}_summary.json")


def default_summary_md_path(args) -> Path:
    rate = str(args.request_rate).replace(".", "p")
    policy = args.scheduler_policy
    if args.scheduler_policy in ("interleave", "unified"):
        policy = f"{policy}_c{args.max_prefill_chunk_tokens}"
    state_path = (
        "statecopy"
        if args.disable_resident_deltanet_state
        else "resident"
    )
    return REPO_ROOT / "docs/qwen35_hybrid" / (
        f"online_{args.backend}_{policy}_{state_path}_{args.workload}_"
        f"r{rate}_summary.md"
    )


def warmup(llm, count: int, vocab_size: int) -> None:
    if count == 0:
        return
    from nanovllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
    for request_id in range(count):
        prompt = [
            1 + ((request_id * 97 + position * 31) % (vocab_size - 1))
            for position in range(128)
        ]
        llm.add_request(prompt, params)
    while not llm.is_finished():
        llm.step()


def _metric_row(name: str, summary: dict[str, float | None]) -> list[object]:
    return [
        name,
        summary.get("mean"),
        summary.get("p50"),
        summary.get("p95"),
        summary.get("p99"),
    ]


def render_markdown(payload: dict[str, object]) -> str:
    config = payload["configuration"]
    summary = payload["summary"]
    workload = payload["workload_metadata"]
    cache_stats = payload.get("cache_stats") or {}
    scheduler_stats = cache_stats.get("scheduler") or {}
    execution_stats = cache_stats.get("execution") or {}
    state_execution_stats = cache_stats.get("state_execution") or {}
    overview = markdown_table(
        ["Metric", "Value"],
        [
            ["backend", config["backend"]],
            [
                "resident DeltaNet state",
                config["resident_deltanet_state"],
            ],
            ["workload", config["workload"]],
            ["scheduler policy", config["scheduler_policy"]],
            [
                "max prefill chunk tokens",
                config["max_prefill_chunk_tokens"],
            ],
            ["max partial prefills", config["max_partial_prefills"]],
            [
                "max long partial prefills",
                config["max_long_partial_prefills"],
            ],
            [
                "long prefill threshold",
                config["long_prefill_token_threshold"],
            ],
            [
                "decode reserve blocks/request",
                config["decode_reserve_blocks_per_seq"],
            ],
            ["request rate", config["request_rate"]],
            ["arrival duration s", summary["arrival_duration_s"]],
            ["wall time including drain s", summary["wall_time_s"]],
            ["requests", summary["request_count"]],
            ["completed", summary["completed_requests"]],
            ["failed", summary["failed_requests"]],
            ["rejected", summary["rejected_requests"]],
            ["incomplete", summary["incomplete_requests"]],
            ["acceptance rate", summary["acceptance_rate"]],
            ["saturated", summary["saturated"]],
            ["input tokens", summary["input_tokens_total"]],
            ["output tokens", summary["output_tokens_total"]],
            ["input tok/s", summary["input_tokens_per_s"]],
            ["output tok/s", summary["output_tokens_per_s"]],
            ["total tok/s", summary["total_tokens_per_s"]],
            ["request throughput req/s", summary["request_throughput_per_s"]],
            ["peak memory GiB", summary["peak_memory_gb"]],
            ["max waiting", summary["max_waiting_requests"]],
            ["max running", summary["max_running_requests"]],
            ["max pending admission queue", summary["max_pending_requests"]],
            ["max inflight", summary["max_inflight_requests"]],
            ["configured max inflight", config["max_inflight_requests"]],
            ["configured max queue", config["max_queue_size"]],
            ["GPU memory utilization", config["gpu_memory_utilization"]],
        ],
    )
    workload_rows = []
    for kind, values in workload["observed_distribution"].items():
        prompt = values["prompt_length"]
        output = values["output_length"]
        workload_rows.append(
            [
                kind,
                values["request_count"],
                prompt["min"],
                prompt["max"],
                prompt["mean"],
                output["min"],
                output["max"],
                output["mean"],
            ]
        )
    workload_table = markdown_table(
        [
            "Workload",
            "Requests",
            "Prompt min",
            "Prompt max",
            "Prompt mean",
            "Output min",
            "Output max",
            "Output mean",
        ],
        workload_rows,
    )
    latency = markdown_table(
        ["Latency", "Mean", "P50", "P95", "P99"],
        [
            _metric_row("TTFT ms", summary["ttft_ms"]),
            _metric_row(
                "Admission delay ms",
                summary["admission_delay_ms"],
            ),
            _metric_row("Scheduler queue time ms", summary["queue_time_ms"]),
            _metric_row("Prefill time ms", summary["prefill_time_ms"]),
            _metric_row("Decode time ms", summary["decode_time_ms"]),
            _metric_row("ITL ms", summary["itl_ms"]),
            _metric_row("E2E latency ms", summary["e2e_latency_ms"]),
        ],
    )
    by_workload = markdown_table(
        ["Workload", "Requests", "TTFT p95 ms", "ITL p95 ms", "E2E p95 ms"],
        [
            [
                kind,
                values["request_count"],
                values["ttft_ms"]["p95"],
                values["itl_ms"]["p95"],
                values["e2e_latency_ms"]["p95"],
            ]
            for kind, values in summary["by_workload"].items()
        ],
    )
    runtime_table = markdown_table(
        ["Runtime metric", "Value"],
        [
            ["ModelRunner calls", execution_stats.get("model_runner_calls")],
            [
                "Unified ModelRunner calls",
                execution_stats.get("unified_model_runner_calls"),
            ],
            ["State gather calls", execution_stats.get("state_gather_calls")],
            ["State commit calls", execution_stats.get("state_commit_calls")],
            [
                "Resident state views",
                execution_stats.get("state_resident_view_calls"),
            ],
            [
                "State allocation rows",
                state_execution_stats.get("allocation_rows"),
            ],
            [
                "State allocation zero ops",
                state_execution_stats.get("allocation_zero_ops"),
            ],
            [
                "Resident commits skipped",
                execution_stats.get("state_commit_skipped_calls"),
            ],
            [
                "Fallback state gathers",
                state_execution_stats.get("fallback_gather_calls"),
            ],
            [
                "Fallback state commits",
                state_execution_stats.get("fallback_commit_calls"),
            ],
            [
                "State compactions",
                state_execution_stats.get("compaction_calls"),
            ],
            [
                "State compaction bytes",
                state_execution_stats.get("compaction_bytes"),
            ],
            [
                "Estimated state copy bytes avoided",
                state_execution_stats.get(
                    "estimated_state_copy_bytes_avoided"
                ),
            ],
            [
                "Max KV block utilization",
                scheduler_stats.get("max_kv_block_utilization"),
            ],
            [
                "Max KV reserved blocks",
                scheduler_stats.get("max_kv_reserved_blocks"),
            ],
            ["Max Delta state slots", cache_stats.get("max_allocated")],
            ["Max Delta state utilization", cache_stats.get("max_utilization")],
        ],
    )
    return f"""# Qwen3.5 Hybrid Online Serving Benchmark

## Methodology

Requests follow a seeded Poisson open-loop arrival process. Arrival timestamps are
fixed before execution, so requests that arrive while a synchronous engine step is
running accumulate queueing delay in TTFT. The benchmark stops generating arrivals
after {summary['arrival_duration_s']:.3f} seconds and then drains all submitted work.

{overview}

## Latency

{latency}

{by_workload}

## Runtime Accounting

{runtime_table}

TTFT is measured from planned arrival to the first generated token. ITL is the interval
between adjacent generated tokens for the same request, beginning with token 2 minus
token 1. Arrival-to-token-1 is TTFT and is never included in ITL. Throughput uses total
wall time including queue drain.

## Reproducible Workload

Arrival process: {workload['arrival_process']}; seed: {workload['seed']}.
Configured workload definition: `{workload['definition']}`.

{workload_table}

Raw per-request timestamps and output tokens are stored in `{payload['raw_trace']}`.
"""


@torch.inference_mode()
def main():
    args = parse_args()
    validate_args(args)
    require_cuda()

    from nanovllm import LLM

    facts = load_model_facts(args.model)
    requests = generate_open_loop_requests(
        request_rate=args.request_rate,
        duration_s=args.duration,
        workload=args.workload,
        vocab_size=int(facts["vocab_size"]),
        seed=args.seed,
        max_requests=args.max_requests,
    )
    max_model_len = max(
        len(request.input_ids) + request.max_output_tokens + 1
        for request in requests
    )
    backend = (
        "chunked" if args.backend == "nanovllm_chunked" else "sequential"
    )
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=args.max_num_seqs,
        hybrid_state_capacity=args.max_inflight_requests,
        resident_deltanet_state=not args.disable_resident_deltanet_state,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        deltanet_backend=backend,
        deltanet_chunk_size=args.deltanet_chunk_size,
        scheduler_policy=args.scheduler_policy,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        max_partial_prefills=args.max_partial_prefills,
        max_long_partial_prefills=args.max_long_partial_prefills,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        decode_reserve_blocks_per_seq=args.decode_reserve_blocks_per_seq,
    )
    metrics = ServingMetrics(requests)
    workload = workload_metadata(
        requests,
        workload=args.workload,
        request_rate=args.request_rate,
        duration_s=args.duration,
        seed=args.seed,
    )
    adapter = NanoVLLMAdapter(llm)
    benchmark_error = None
    benchmark_started = None
    wall_time_s = 0.0
    cache_stats = None
    measured_peak_memory_gb = None
    try:
        warmup(llm, args.warmup_requests, int(facts["vocab_size"]))
        llm.model_runner.call("reset_execution_stats")
        llm.scheduler.reset_resource_stats()
        reset_peak_memory()

        def progress(message: str) -> None:
            if not args.quiet:
                print(f"[serving] {message}", flush=True)

        benchmark_started = perf_counter()
        wall_time_s = run_online_workload(
            adapter=adapter,
            requests=requests,
            metrics=metrics,
            progress=progress,
            arrival_window_s=(
                requests[-1].arrival_time
                if args.max_requests is not None
                and len(requests) == args.max_requests
                else args.duration
            ),
            max_queue_size=args.max_queue_size,
            max_inflight_requests=args.max_inflight_requests,
        )
        cache_stats = llm.model_runner.call("get_hybrid_state_stats")
        cache_stats["scheduler"] = llm.scheduler.get_resource_stats()
        cache_stats["execution"] = llm.model_runner.call(
            "get_execution_stats"
        )
    except Exception as exc:
        benchmark_error = f"{type(exc).__name__}: {exc}"
        wall_time_s = (
            perf_counter() - benchmark_started
            if benchmark_started is not None
            else 0.0
        )
        metrics.fail_submitted_requests(wall_time_s, benchmark_error)
        try:
            cache_stats = llm.model_runner.call("get_hybrid_state_stats")
            cache_stats["scheduler"] = llm.scheduler.get_resource_stats()
            cache_stats["execution"] = llm.model_runner.call(
                "get_execution_stats"
            )
        except Exception:
            cache_stats = None
    finally:
        try:
            if torch.cuda.is_available():
                measured_peak_memory_gb = peak_memory_gb()
        except Exception:
            measured_peak_memory_gb = None
        try:
            llm.exit()
        except Exception as exc:
            if benchmark_error is None:
                benchmark_error = f"{type(exc).__name__}: {exc}"

    effective_arrival_duration = (
        requests[-1].arrival_time
        if args.max_requests is not None
        and len(requests) == args.max_requests
        else args.duration
    )
    summary = metrics.summarize(
        wall_time_s=wall_time_s,
        arrival_duration_s=effective_arrival_duration,
        peak_memory_gb=measured_peak_memory_gb,
    )
    json_path = Path(args.save_json) if args.save_json else default_json_path(args)
    summary_json_path = (
        Path(args.save_summary_json)
        if args.save_summary_json
        else default_summary_json_path(json_path)
    )
    summary_md_path = (
        Path(args.save_summary_md or args.save_md)
        if args.save_summary_md or args.save_md
        else default_summary_md_path(args)
    )
    payload = {
        "environment": environment_metadata(args.model),
        "model_facts": facts,
        "configuration": {
            "backend": args.backend,
            "workload": args.workload,
            "request_rate": args.request_rate,
            "duration_s": args.duration,
            "seed": args.seed,
            "max_requests": args.max_requests,
            "max_num_seqs": args.max_num_seqs,
            "max_inflight_requests": args.max_inflight_requests,
            "max_queue_size": args.max_queue_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "deltanet_chunk_size": args.deltanet_chunk_size,
            "resident_deltanet_state": (
                not args.disable_resident_deltanet_state
            ),
            "scheduler_policy": args.scheduler_policy,
            "max_prefill_chunk_tokens": args.max_prefill_chunk_tokens,
            "max_partial_prefills": args.max_partial_prefills,
            "max_long_partial_prefills": args.max_long_partial_prefills,
            "long_prefill_token_threshold": args.long_prefill_token_threshold,
            "decode_reserve_blocks_per_seq": args.decode_reserve_blocks_per_seq,
            "warmup_requests": args.warmup_requests,
        },
        "workload_metadata": workload,
        "summary": summary,
        "cache_stats": cache_stats,
        "benchmark_error": benchmark_error,
        "artifacts": {
            "raw_trace": str(json_path),
            "summary_json": str(summary_json_path),
            "summary_markdown": str(summary_md_path),
        },
        "requests": metrics.request_records(),
    }
    summary_payload = {
        "environment": payload["environment"],
        "model_facts": facts,
        "configuration": payload["configuration"],
        "workload_metadata": workload,
        "summary": summary,
        "cache_stats": cache_stats,
        "benchmark_error": benchmark_error,
        "raw_trace": str(json_path),
    }
    write_json(json_path, payload)
    write_json(summary_json_path, summary_payload)
    write_text(summary_md_path, render_markdown(summary_payload))
    print(f"Saved {json_path}", flush=True)
    print(f"Saved {summary_json_path}", flush=True)
    print(f"Saved {summary_md_path}", flush=True)
    if benchmark_error is not None:
        raise RuntimeError(
            f"Serving benchmark failed after writing partial artifacts: "
            f"{benchmark_error}"
        )


if __name__ == "__main__":
    main()
