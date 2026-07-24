#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.bench_e2e import run_nanovllm_case  # noqa: E402
from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    deterministic_prompts,
    environment_metadata,
    load_model_facts,
    markdown_table,
    parse_int_list,
    require_cuda,
    safe_error,
    summarize,
    write_json,
    write_text,
)


METRICS = (
    "elapsed_s",
    "ttft_s",
    "average_itl_ms",
    "p50_itl_ms",
    "p95_itl_ms",
    "decode_tokens_per_s",
    "e2e_tokens_per_s",
    "peak_memory_gb",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="A/B benchmark Qwen3.5 normal decode preparation vs cached fast path."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", default="1,4,8,16")
    parser.add_argument("--prompt-lens", default="512,2048")
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
        help="Leave headroom for Qwen3.5 prefill temporaries.",
    )
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "decode_fast_path_ab.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/10_decode_fast_path_results.md"),
    )
    parser.add_argument(
        "--worker-mode",
        choices=("normal", "fast"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_worker(args):
    require_cuda()
    from nanovllm import LLM

    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_lens = parse_int_list(args.prompt_lens)
    facts = load_model_facts(args.model)
    fast_path = args.worker_mode == "fast"
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=max(batch_sizes),
        hybrid_state_capacity=max(batch_sizes),
        max_model_len=max(prompt_lens) + args.output_len + 1,
        max_num_batched_tokens=max(batch_sizes) * max(prompt_lens),
        gpu_memory_utilization=args.gpu_memory_utilization,
        resident_deltanet_state=True,
        decode_fast_path=fast_path,
        deltanet_backend="chunked",
        deltanet_chunk_size=args.deltanet_chunk_size,
    )
    rows = []
    failures = []
    try:
        for prompt_len in prompt_lens:
            for batch_size in batch_sizes:
                torch.cuda.empty_cache()
                label = (
                    f"mode={args.worker_mode} batch={batch_size} "
                    f"prompt={prompt_len} output={args.output_len}"
                )
                print(f"[decode-fast-path] {label}", flush=True)
                prompts = deterministic_prompts(
                    int(facts["vocab_size"]),
                    batch_size,
                    prompt_len,
                )
                try:
                    for _ in range(args.warmup):
                        run_nanovllm_case(llm, prompts, args.output_len)
                    for repeat_idx in range(args.repeat):
                        llm.model_runner.call("reset_execution_stats")
                        metrics = run_nanovllm_case(
                            llm,
                            prompts,
                            args.output_len,
                        )
                        execution_stats = llm.model_runner.call(
                            "get_execution_stats"
                        )
                        rows.append(
                            {
                                "mode": args.worker_mode,
                                "decode_fast_path": fast_path,
                                "batch_size": batch_size,
                                "prompt_len": prompt_len,
                                "output_len": args.output_len,
                                "repeat": repeat_idx,
                                **metrics,
                                "execution_stats": execution_stats,
                            }
                        )
                    torch.cuda.empty_cache()
                except Exception as exc:
                    failures.append(
                        {
                            "mode": args.worker_mode,
                            "batch_size": batch_size,
                            "prompt_len": prompt_len,
                            "error": safe_error(exc),
                        }
                    )
                    print(
                        f"[decode-fast-path] FAILED {label}: {exc}",
                        flush=True,
                    )
                    return {
                        "mode": args.worker_mode,
                        "rows": rows,
                        "failures": failures,
                        "aborted_after_failure": True,
                    }
    finally:
        llm.exit()
    return {
        "mode": args.worker_mode,
        "rows": rows,
        "failures": failures,
    }


def invoke_worker(args, mode, output_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--batch-sizes",
        args.batch_sizes,
        "--prompt-lens",
        args.prompt_lens,
        "--output-len",
        str(args.output_len),
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--deltanet-chunk-size",
        str(args.deltanet_chunk_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--worker-mode",
        mode,
        "--worker-output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH",
        "",
    )
    print(f"[decode-fast-path] starting isolated {mode} worker", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
    if not output_path.exists():
        return {
            "mode": mode,
            "rows": [],
            "failures": [
                {
                    "mode": mode,
                    "error": {
                        "type": "WorkerFailure",
                        "message": f"worker exited with code {completed.returncode}",
                    },
                }
            ],
        }
    return json.loads(output_path.read_text())


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (row["mode"], row["batch_size"], row["prompt_len"])
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = {
            "mode": key[0],
            "batch_size": key[1],
            "prompt_len": key[2],
            "output_len": group[0]["output_len"],
            "repeats": len(group),
        }
        for metric in METRICS:
            for statistic, value in summarize(
                row.get(metric) for row in group
            ).items():
                summary[f"{metric}_{statistic}"] = value
        stat_names = (
            "decode_fast_path_hits",
            "decode_fast_path_normal_steps",
            "decode_fast_path_builds",
            "decode_fast_path_invalidations",
            "decode_fast_path_state_view_reuses",
        )
        for name in stat_names:
            summary[f"{name}_mean"] = summarize(
                row["execution_stats"].get(name, 0) for row in group
            )["mean"]
        summaries.append(summary)
    return summaries


def compare_modes(summaries):
    grouped = {}
    for row in summaries:
        key = (row["batch_size"], row["prompt_len"], row["output_len"])
        grouped.setdefault(key, {})[row["mode"]] = row
    comparisons = []
    for key, modes in sorted(grouped.items()):
        normal = modes.get("normal")
        fast = modes.get("fast")
        if normal is None or fast is None:
            continue
        comparisons.append(
            {
                "batch_size": key[0],
                "prompt_len": key[1],
                "output_len": key[2],
                "decode_throughput_ratio": (
                    fast["decode_tokens_per_s_mean"]
                    / normal["decode_tokens_per_s_mean"]
                ),
                "average_itl_ratio": (
                    normal["average_itl_ms_mean"]
                    / fast["average_itl_ms_mean"]
                ),
                "p95_itl_ratio": (
                    normal["p95_itl_ms_mean"]
                    / fast["p95_itl_ms_mean"]
                ),
                "fast_path_hits_mean": fast[
                    "decode_fast_path_hits_mean"
                ],
                "fast_path_invalidations_mean": fast[
                    "decode_fast_path_invalidations_mean"
                ],
            }
        )
    return comparisons


def render_markdown(payload):
    results = markdown_table(
        [
            "mode",
            "batch",
            "prompt",
            "ITL mean ms",
            "ITL p50 ms",
            "ITL p95 ms",
            "decode tok/s",
            "fast hits",
            "invalidations",
        ],
        [
            [
                row["mode"],
                row["batch_size"],
                row["prompt_len"],
                row["average_itl_ms_mean"],
                row["p50_itl_ms_mean"],
                row["p95_itl_ms_mean"],
                row["decode_tokens_per_s_mean"],
                row["decode_fast_path_hits_mean"],
                row["decode_fast_path_invalidations_mean"],
            ]
            for row in payload["summary"]
        ],
    )
    comparison = markdown_table(
        [
            "batch",
            "prompt",
            "decode throughput ratio",
            "mean ITL speedup",
            "p95 ITL speedup",
            "fast hits",
            "invalidations",
        ],
        [
            [
                row["batch_size"],
                row["prompt_len"],
                row["decode_throughput_ratio"],
                row["average_itl_ratio"],
                row["p95_itl_ratio"],
                row["fast_path_hits_mean"],
                row["fast_path_invalidations_mean"],
            ]
            for row in payload["comparison"]
        ],
    )
    failures = payload["failures"]
    failure_text = (
        "None."
        if not failures
        else "\n".join(
            f"- `{row['mode']}` batch={row.get('batch_size', 'N/A')} "
            f"prompt={row.get('prompt_len', 'N/A')}: {row['error']}"
            for row in failures
        )
    )
    return f"""# Qwen3.5 Decode Fast Path A/B

Qwen3.5-9B BF16 eager decode with identical deterministic prompts, resident
DeltaNet state, chunked prefill, warmup={payload['matrix']['warmup']}, and
repeat={payload['matrix']['repeat']}. GPU memory utilization is
{payload['matrix']['gpu_memory_utilization']}. Workers run in isolated processes.

## Results

{results}

## Fast / Normal

{comparison}

The fast path caches request/layout identity, typed attention metadata, resident
state views, and stable device buffers. It falls back to normal preparation when
the active batch, state-slot layout, or KV block-table layout changes. This table
does not claim a kernel optimization.

## Failures

{failure_text}
"""


def main():
    args = parse_args()
    if args.output_len < 2 or args.warmup < 0 or args.repeat < 1:
        raise ValueError(
            "--output-len must be >= 2, --warmup >= 0, and --repeat >= 1"
        )
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.worker_mode:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        try:
            payload = run_worker(args)
        except Exception as exc:
            payload = {
                "mode": args.worker_mode,
                "rows": [],
                "failures": [
                    {"mode": args.worker_mode, "error": safe_error(exc)}
                ],
            }
        write_json(args.worker_output, payload)
        return

    require_cuda()
    worker_dir = DEFAULT_RESULTS_DIR / ".decode_fast_path_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    paths = []
    try:
        for mode in ("normal", "fast"):
            output_path = worker_dir / f"{mode}.json"
            paths.append(output_path)
            outputs.append(invoke_worker(args, mode, output_path))
        rows = [row for output in outputs for row in output.get("rows", [])]
        failures = [
            row
            for output in outputs
            for row in output.get("failures", [])
        ]
        summary = aggregate_rows(rows)
        payload = {
            "schema_version": 1,
            "environment": environment_metadata(args.model),
            "model_facts": load_model_facts(args.model),
            "matrix": {
                "batch_sizes": parse_int_list(args.batch_sizes),
                "prompt_lens": parse_int_list(args.prompt_lens),
                "output_len": args.output_len,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "deltanet_chunk_size": args.deltanet_chunk_size,
                "gpu_memory_utilization": args.gpu_memory_utilization,
            },
            "runs": rows,
            "summary": summary,
            "comparison": compare_modes(summary),
            "failures": failures,
        }
        write_json(args.save_json, payload)
        write_text(args.save_md, render_markdown(payload))
        print(f"Saved {args.save_json}", flush=True)
        print(f"Saved {args.save_md}", flush=True)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
        try:
            worker_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
