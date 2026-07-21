#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    deterministic_prompts,
    environment_metadata,
    load_hf_text_reference,
    load_model_facts,
    markdown_table,
    parse_int_list,
    peak_memory_gb,
    percentile,
    require_cuda,
    reset_peak_memory,
    safe_error,
    summarize,
    synchronize,
    theoretical_cache_bytes,
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
        description="Benchmark Qwen3.5 HF eager, nano-vLLM hybrid, and vLLM."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--backends", default="hf,nanovllm,vllm")
    parser.add_argument("--batch-sizes", default="1,2,4,8")
    parser.add_argument("--prompt-lens", default="128,512,2048")
    parser.add_argument("--output-lens", default="32,128")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "e2e_benchmark.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/04_benchmark_results.md"),
    )
    parser.add_argument(
        "--worker",
        choices=(
            "hf",
            "nanovllm",
            "nanovllm_sequential",
            "nanovllm_chunked",
            "vllm",
        ),
    )
    parser.add_argument("--worker-output")
    return parser.parse_args()


def step_statistics(step_times_s):
    milliseconds = [value * 1000 for value in step_times_s]
    return {
        "average_itl_ms": sum(milliseconds) / len(milliseconds)
        if milliseconds
        else None,
        "p50_itl_ms": percentile(milliseconds, 0.5),
        "p95_itl_ms": percentile(milliseconds, 0.95),
    }


@torch.inference_mode()
def run_hf_case(model, prompts, output_len):
    input_ids = torch.tensor(prompts, device="cuda")
    reset_peak_memory()
    synchronize()
    start = time.perf_counter()
    outputs = model(input_ids=input_ids, use_cache=True)
    tokens = outputs.logits[:, -1].argmax(-1)
    synchronize()
    ttft_s = time.perf_counter() - start
    cache = outputs.past_key_values
    decode_steps = []
    for _ in range(1, output_len):
        synchronize()
        step_start = time.perf_counter()
        outputs = model(
            input_ids=tokens.unsqueeze(1),
            past_key_values=cache,
            use_cache=True,
        )
        tokens = outputs.logits[:, -1].argmax(-1)
        cache = outputs.past_key_values
        synchronize()
        decode_steps.append(time.perf_counter() - step_start)
    elapsed_s = ttft_s + sum(decode_steps)
    batch_size = len(prompts)
    decode_tokens = batch_size * max(0, output_len - 1)
    decode_time = sum(decode_steps)
    row = {
        "elapsed_s": elapsed_s,
        "ttft_s": ttft_s,
        "decode_tokens_per_s": decode_tokens / decode_time
        if decode_time
        else None,
        "e2e_tokens_per_s": batch_size * output_len / elapsed_s,
        "peak_memory_gb": peak_memory_gb(),
        **step_statistics(decode_steps),
    }
    del outputs, cache, tokens, input_ids
    return row


def run_nanovllm_case(llm, prompts, output_len):
    from nanovllm import SamplingParams

    params = SamplingParams(
        temperature=0.0, max_tokens=output_len, ignore_eos=True
    )
    for prompt in prompts:
        llm.add_request(prompt, params)
    reset_peak_memory()
    synchronize()
    total_start = time.perf_counter()
    prefill_time = 0.0
    decode_steps = []
    while not llm.is_finished():
        synchronize()
        step_start = time.perf_counter()
        _, num_tokens = llm.step()
        synchronize()
        step_time = time.perf_counter() - step_start
        if num_tokens > 0:
            prefill_time += step_time
        else:
            decode_steps.append(step_time)
    elapsed_s = time.perf_counter() - total_start
    batch_size = len(prompts)
    decode_tokens = batch_size * max(0, output_len - 1)
    decode_time = sum(decode_steps)
    return {
        "elapsed_s": elapsed_s,
        "ttft_s": prefill_time,
        "decode_tokens_per_s": decode_tokens / decode_time
        if decode_time
        else None,
        "e2e_tokens_per_s": batch_size * output_len / elapsed_s,
        "peak_memory_gb": peak_memory_gb(),
        **step_statistics(decode_steps),
    }


def _metric_value(metrics, name):
    value = getattr(metrics, name, None) if metrics is not None else None
    return float(value) if value is not None else None


def run_vllm_case(llm, prompts, output_len):
    from vllm import SamplingParams as VllmSamplingParams

    inputs = [{"prompt_token_ids": prompt} for prompt in prompts]
    params = VllmSamplingParams(
        temperature=0.0, max_tokens=output_len, ignore_eos=True
    )
    reset_peak_memory()
    synchronize()
    start = time.perf_counter()
    outputs = llm.generate(inputs, params, use_tqdm=False)
    synchronize()
    elapsed_s = time.perf_counter() - start

    ttfts = []
    average_itls = []
    for output in outputs:
        metrics = getattr(output, "metrics", None)
        arrival = _metric_value(metrics, "arrival_time")
        first = _metric_value(metrics, "first_token_time")
        finished = _metric_value(metrics, "finished_time")
        if arrival is not None and first is not None:
            ttfts.append(first - arrival)
        if first is not None and finished is not None and output_len > 1:
            average_itls.append((finished - first) * 1000 / (output_len - 1))

    batch_size = len(prompts)
    generated = sum(
        len(candidate.token_ids)
        for output in outputs
        for candidate in output.outputs[:1]
    )
    ttft_s = sum(ttfts) / len(ttfts) if ttfts else None
    average_itl_ms = (
        sum(average_itls) / len(average_itls) if average_itls else None
    )
    decode_time = (
        average_itl_ms / 1000 * (output_len - 1)
        if average_itl_ms is not None
        else None
    )
    return {
        "elapsed_s": elapsed_s,
        "ttft_s": ttft_s,
        "average_itl_ms": average_itl_ms,
        "p50_itl_ms": percentile(average_itls, 0.5),
        "p95_itl_ms": percentile(average_itls, 0.95),
        "decode_tokens_per_s": batch_size * (output_len - 1) / decode_time
        if decode_time
        else None,
        "e2e_tokens_per_s": generated / elapsed_s,
        "peak_memory_gb": peak_memory_gb(),
    }


def create_backend(args, facts):
    max_batch = max(parse_int_list(args.batch_sizes))
    max_prompt = max(parse_int_list(args.prompt_lens))
    max_output = max(parse_int_list(args.output_lens))
    max_model_len = max_prompt + max_output + 1
    if args.worker == "hf":
        model, _, _ = load_hf_text_reference(args.model)
        return model, run_hf_case, None
    if args.worker.startswith("nanovllm"):
        from nanovllm import LLM

        deltanet_backend = (
            "chunked" if args.worker == "nanovllm_chunked" else "sequential"
        )
        llm = LLM(
            args.model,
            enforce_eager=True,
            max_num_seqs=max_batch,
            hybrid_state_capacity=max_batch,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_batch * max_prompt,
            deltanet_backend=deltanet_backend,
            deltanet_chunk_size=args.deltanet_chunk_size,
        )
        stats = llm.model_runner.call("get_hybrid_state_stats")
        return llm, run_nanovllm_case, stats
    try:
        from vllm import LLM as VllmLLM
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed in this environment") from exc
    llm = VllmLLM(
        model=args.model,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=max_model_len,
        max_num_seqs=max_batch,
        trust_remote_code=False,
    )
    return llm, run_vllm_case, None


def run_backend_worker(args, batch_sizes, prompt_lens, output_lens):
    require_cuda()
    facts = load_model_facts(args.model)
    rows = []
    failures = []
    backend = runner = cache_stats = None
    try:
        backend, runner, cache_stats = create_backend(args, facts)
        for batch_size in batch_sizes:
            for prompt_len in prompt_lens:
                prompts = deterministic_prompts(
                    int(facts["vocab_size"]), batch_size, prompt_len
                )
                for output_len in output_lens:
                    label = (
                        f"backend={args.worker} batch={batch_size} "
                        f"prompt={prompt_len} output={output_len}"
                    )
                    print(f"[e2e] {label}", flush=True)
                    try:
                        for _ in range(args.warmup):
                            runner(backend, prompts, output_len)
                        for repeat_idx in range(args.repeat):
                            metrics = runner(backend, prompts, output_len)
                            active_kv_bytes, active_delta_bytes = theoretical_cache_bytes(
                                facts,
                                batch_size,
                                prompt_len + max(0, output_len - 1),
                            )
                            rows.append(
                                {
                                    "backend": args.worker,
                                    "batch_size": batch_size,
                                    "prompt_len": prompt_len,
                                    "output_len": output_len,
                                    "repeat": repeat_idx,
                                    "active_kv_cache_bytes": active_kv_bytes,
                                    "active_delta_state_bytes": active_delta_bytes,
                                    "engine_kv_cache_bytes": cache_stats.get(
                                        "kv_cache_bytes"
                                    )
                                    if cache_stats
                                    else None,
                                    "engine_delta_pool_bytes": cache_stats.get(
                                        "delta_pool_bytes"
                                    )
                                    if cache_stats
                                    else None,
                                    **metrics,
                                }
                            )
                    except Exception as exc:
                        failures.append(
                            {
                                "backend": args.worker,
                                "batch_size": batch_size,
                                "prompt_len": prompt_len,
                                "output_len": output_len,
                                "error": safe_error(exc),
                            }
                        )
                        print(f"[e2e] FAILED {label}: {exc}", flush=True)
                        torch.cuda.empty_cache()
                        return {
                            "backend": args.worker,
                            "rows": rows,
                            "failures": failures,
                            "cache_stats": cache_stats,
                            "aborted_after_failure": True,
                        }
    finally:
        if args.worker.startswith("nanovllm") and backend is not None:
            backend.exit()
    return {
        "backend": args.worker,
        "rows": rows,
        "failures": failures,
        "cache_stats": cache_stats,
    }


def invoke_backend(args, backend, output_path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--backends",
        args.backends,
        "--batch-sizes",
        args.batch_sizes,
        "--prompt-lens",
        args.prompt_lens,
        "--output-lens",
        args.output_lens,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--deltanet-chunk-size",
        str(args.deltanet_chunk_size),
        "--worker",
        backend,
        "--worker-output",
        str(output_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    print(f"[e2e] starting isolated {backend} worker", flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, env=environment)
    if output_path.exists():
        payload = __import__("json").loads(output_path.read_text())
    else:
        payload = {
            "backend": backend,
            "rows": [],
            "failures": [
                {
                    "backend": backend,
                    "error": {
                        "type": "WorkerFailure",
                        "message": f"worker exited with code {completed.returncode}",
                    },
                }
            ],
        }
    return payload


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = (
            row["backend"],
            row["batch_size"],
            row["prompt_len"],
            row["output_len"],
        )
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, group in sorted(groups.items()):
        summary = {
            "backend": key[0],
            "batch_size": key[1],
            "prompt_len": key[2],
            "output_len": key[3],
            "repeats": len(group),
            "active_kv_cache_bytes": group[0]["active_kv_cache_bytes"],
            "active_delta_state_bytes": group[0]["active_delta_state_bytes"],
            "engine_kv_cache_bytes": group[0]["engine_kv_cache_bytes"],
            "engine_delta_pool_bytes": group[0]["engine_delta_pool_bytes"],
        }
        for metric in METRICS:
            stats = summarize(row.get(metric) for row in group)
            for statistic, value in stats.items():
                summary[f"{metric}_{statistic}"] = value
        summaries.append(summary)
    return summaries


def compare_deltanet_backends(summaries):
    grouped = {}
    for row in summaries:
        key = (row["batch_size"], row["prompt_len"], row["output_len"])
        grouped.setdefault(key, {})[row["backend"]] = row
    comparisons = []
    for key, backends in sorted(grouped.items()):
        sequential = backends.get("nanovllm_sequential")
        chunked = backends.get("nanovllm_chunked")
        if sequential is None or chunked is None:
            continue
        sequential_ttft = sequential["ttft_s_mean"]
        chunked_ttft = chunked["ttft_s_mean"]
        sequential_decode = sequential["decode_tokens_per_s_mean"]
        chunked_decode = chunked["decode_tokens_per_s_mean"]
        sequential_peak = sequential["peak_memory_gb_mean"]
        chunked_peak = chunked["peak_memory_gb_mean"]
        comparisons.append(
            {
                "batch_size": key[0],
                "prompt_len": key[1],
                "output_len": key[2],
                "ttft_speedup": sequential_ttft / chunked_ttft
                if chunked_ttft
                else None,
                "decode_throughput_ratio": chunked_decode / sequential_decode
                if sequential_decode
                else None,
                "peak_memory_delta_gb": chunked_peak - sequential_peak
                if chunked_peak is not None and sequential_peak is not None
                else None,
            }
        )
    return comparisons


def render_markdown(payload):
    rows = payload["summary"]
    table = markdown_table(
        [
            "backend",
            "batch",
            "prompt",
            "output",
            "TTFT mean ms",
            "ITL mean ms",
            "ITL p95 ms",
            "decode tok/s",
            "E2E tok/s",
            "peak GiB",
        ],
        [
            [
                row["backend"],
                row["batch_size"],
                row["prompt_len"],
                row["output_len"],
                row["ttft_s_mean"] * 1000
                if row["ttft_s_mean"] is not None
                else None,
                row["average_itl_ms_mean"],
                row["p95_itl_ms_mean"],
                row["decode_tokens_per_s_mean"],
                row["e2e_tokens_per_s_mean"],
                row["peak_memory_gb_mean"],
            ]
            for row in rows
        ],
    )
    memory = markdown_table(
        ["backend", "batch", "processed context", "active KV GiB", "active Delta GiB"],
        [
            [
                row["backend"],
                row["batch_size"],
                row["prompt_len"] + max(0, row["output_len"] - 1),
                row["active_kv_cache_bytes"] / 2**30,
                row["active_delta_state_bytes"] / 2**30,
            ]
            for row in rows
        ],
    )
    comparison = markdown_table(
        [
            "batch",
            "prompt",
            "output",
            "TTFT speedup",
            "decode throughput ratio",
            "chunked - sequential peak GiB",
        ],
        [
            [
                row["batch_size"],
                row["prompt_len"],
                row["output_len"],
                row["ttft_speedup"],
                row["decode_throughput_ratio"],
                row["peak_memory_delta_gb"],
            ]
            for row in payload.get("deltanet_backend_comparison", [])
        ],
    )
    failures = payload["failures"]
    failure_text = (
        "None."
        if not failures
        else "\n".join(
            f"- `{failure.get('backend')}` batch={failure.get('batch_size', 'N/A')} "
            f"prompt={failure.get('prompt_len', 'N/A')} output={failure.get('output_len', 'N/A')}: "
            f"{failure.get('error')}"
            for failure in failures
        )
    )
    return f"""# Qwen3.5 Hybrid E2E Benchmark

Backends run in isolated processes with identical token IDs, BF16 weights, greedy
generation, eager execution, warmup={payload['matrix']['warmup']}, and
repeat={payload['matrix']['repeat']}. `N/A` means that a backend version did not expose
the required request-level timing rather than an inferred value being substituted.
`nanovllm_sequential` and `nanovllm_chunked` differ only in the PyTorch DeltaNet
prefill recurrence execution model; single-token decode remains recurrent.

## Performance

{table}

The JSON artifact contains mean, p50, and p95 across repeats for every metric.

## Sequential -> Chunked DeltaNet

{comparison}

TTFT is the expected impact surface because chunked recurrence is used for prefill.
Single-token decode intentionally keeps the sequential recurrent update, so the decode
throughput ratio is a regression guard rather than the optimization claim.

## Active Cache Footprint

{memory}

Active KV bytes count only full-attention layers. DeltaNet state bytes are reported
separately and use FP32 recurrent matrices.

## Failed Configurations

{failure_text}
"""


def main():
    args = parse_args()
    batch_sizes = parse_int_list(args.batch_sizes)
    prompt_lens = parse_int_list(args.prompt_lens)
    output_lens = parse_int_list(args.output_lens)
    if args.repeat < 1 or args.warmup < 0 or args.deltanet_chunk_size <= 0:
        raise ValueError("--repeat must be >= 1 and --warmup must be >= 0")
    if args.worker:
        if not args.worker_output:
            raise ValueError("--worker-output is required in worker mode")
        try:
            payload = run_backend_worker(
                args, batch_sizes, prompt_lens, output_lens
            )
        except Exception as exc:
            payload = {
                "backend": args.worker,
                "rows": [],
                "failures": [
                    {"backend": args.worker, "error": safe_error(exc)}
                ],
            }
        write_json(args.worker_output, payload)
        return

    require_cuda()
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    unsupported = set(backends) - {
        "hf",
        "nanovllm",
        "nanovllm_sequential",
        "nanovllm_chunked",
        "vllm",
    }
    if unsupported:
        raise ValueError(f"Unsupported backends: {sorted(unsupported)}")
    worker_dir = DEFAULT_RESULTS_DIR / ".e2e_workers"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_outputs = []
    paths = []
    try:
        for backend in backends:
            path = worker_dir / f"{backend}.json"
            paths.append(path)
            worker_outputs.append(invoke_backend(args, backend, path))
        rows = [row for output in worker_outputs for row in output.get("rows", [])]
        failures = [
            failure
            for output in worker_outputs
            for failure in output.get("failures", [])
        ]
        summary = aggregate_rows(rows)
        payload = {
            "environment": environment_metadata(args.model),
            "model_facts": load_model_facts(args.model),
            "matrix": {
                "backends": backends,
                "batch_sizes": batch_sizes,
                "prompt_lens": prompt_lens,
                "output_lens": output_lens,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "deltanet_chunk_size": args.deltanet_chunk_size,
            },
            "runs": rows,
            "summary": summary,
            "deltanet_backend_comparison": compare_deltanet_backends(summary),
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
