import argparse
import json
from pathlib import Path
import subprocess
from statistics import mean
from time import perf_counter

import torch


def make_token_prompt(llm, prompt_len: int) -> list[int]:
    token_ids = llm.tokenizer.encode(" benchmark", add_special_tokens=False)
    token_id = token_ids[0] if token_ids else 0
    eos = llm.tokenizer.eos_token_id
    if token_id == eos:
        token_id = 0 if eos != 0 else 1
    return [token_id] * prompt_len


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(row: dict) -> str:
    keys = list(row.keys())
    return "\n".join([
        "| Metric | Value |",
        "|---|---:|",
        *[f"| {key} | {format_value(row[key])} |" for key in keys],
    ])


def markdown_rows(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def aggregate_rows(rows: list[dict], keys: list[str]) -> list[dict]:
    aggregate = []
    for key in keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        aggregate.append({
            "metric": key,
            "mean": mean(values),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
        })
    return aggregate


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run_profiled_generation(llm, prompts: list[list[int]], sampling_params):
    outputs = {}
    prefill_time_s = 0.0
    decode_time_s = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    decode_step_latencies = []
    first_decode_end = None

    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    torch.cuda.synchronize()
    start = perf_counter()
    while not llm.is_finished():
        torch.cuda.synchronize()
        step_start = perf_counter()
        output, num_tokens = llm.step()
        torch.cuda.synchronize()
        step_elapsed = perf_counter() - step_start

        if num_tokens > 0:
            prefill_time_s += step_elapsed
            prefill_tokens += num_tokens
        else:
            step_decode_tokens = -num_tokens
            decode_time_s += step_elapsed
            decode_tokens += step_decode_tokens
            decode_step_latencies.append(step_elapsed)
            if first_decode_end is None:
                first_decode_end = perf_counter()

        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    ordered_outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
    return {
        "outputs": ordered_outputs,
        "elapsed_s": elapsed,
        "ttft_s": (first_decode_end - start) if first_decode_end is not None else 0.0,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "decode_tokens_per_s": decode_tokens / decode_time_s if decode_time_s else 0.0,
        "total_tokens_per_s": (prefill_tokens + decode_tokens) / elapsed if elapsed else 0.0,
        "itl_ms_avg": (decode_time_s / decode_tokens * 1000.0) if decode_tokens else 0.0,
        "decode_step_ms_p50": percentile(decode_step_latencies, 50) * 1000.0,
        "decode_step_ms_p95": percentile(decode_step_latencies, 95) * 1000.0,
        "decode_steps": len(decode_step_latencies),
    }


def run_once(args, run_index: int, LLM, SamplingParams):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    llm = None
    try:
        llm = LLM(
            args.model,
            max_model_len=args.prompt_len + args.max_tokens,
            max_num_seqs=args.num_prompts,
            enforce_eager=args.enforce_eager,
            attn_backend=args.attn_backend,
            kvcache_block_size=args.block_size,
            triton_paged_decode_auto_threshold=args.auto_threshold,
        )
        prompt = make_token_prompt(llm, args.prompt_len)
        prompts = [prompt[:] for _ in range(args.num_prompts)]
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        result = run_profiled_generation(llm, prompts, sampling_params)
        generated_tokens = sum(len(token_ids) for token_ids in result["outputs"])
        prompt_tokens = args.prompt_len * args.num_prompts
        total_tokens = prompt_tokens + generated_tokens
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        metrics = llm.metrics()
        kv_dtype = metrics["kv_cache_dtype"]
        return {
            "run": run_index,
            "git_commit": get_git_commit(),
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "model_dtype": metrics.get("model_dtype", kv_dtype),
            "prompt_len": args.prompt_len,
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "max_new_tokens": args.max_tokens,
            "elapsed_s": result["elapsed_s"],
            "ttft_s": result["ttft_s"],
            "prefill_time_s": result["prefill_time_s"],
            "decode_time_s": result["decode_time_s"],
            "decode_steps": result["decode_steps"],
            "prompt_tokens": prompt_tokens,
            "profiled_prefill_tokens": result["prefill_tokens"],
            "generated_tokens": generated_tokens,
            "profiled_decode_tokens": result["decode_tokens"],
            "total_tokens_per_s": total_tokens / result["elapsed_s"],
            "profiled_total_tokens_per_s": result["total_tokens_per_s"],
            "decode_tokens_per_s": result["decode_tokens_per_s"],
            "itl_ms_avg": result["itl_ms_avg"],
            "decode_step_ms_p50": result["decode_step_ms_p50"],
            "decode_step_ms_p95": result["decode_step_ms_p95"],
            "peak_gpu_memory_gb": peak_mem_gb,
            "num_kvcache_blocks": metrics["num_kvcache_blocks"],
            "used_blocks": metrics["used_blocks"],
            "free_blocks": metrics["free_blocks"],
            "block_utilization": metrics["block_utilization"],
            "max_used_blocks": metrics["max_used_blocks"],
            "max_block_utilization": metrics["max_block_utilization"],
            "active_sequences": metrics.get("active_sequences", 0),
            "allocated_blocks_per_sequence": metrics.get("allocated_blocks_per_sequence", {}),
            "prefix_cache_hits": metrics["prefix_cache_hits"],
            "prefix_cache_misses": metrics["prefix_cache_misses"],
            "prefix_cache_hit_rate": metrics["prefix_cache_hit_rate"],
            "kv_cache_dtype": kv_dtype,
            "resolved_kv_cache_dtype": kv_dtype,
            "max_num_seqs": metrics["max_num_seqs"],
            "max_model_len": metrics["max_model_len"],
            "norm_backend": metrics["norm_backend"],
            "activation_backend": metrics["activation_backend"],
            "rope_backend": metrics["rope_backend"],
            "linear_backend": metrics["linear_backend"],
            "attn_backend": metrics["attn_backend"],
            "kvcache_block_size": metrics["kvcache_block_size"],
            "triton_paged_decode_auto_threshold": metrics["triton_paged_decode_auto_threshold"],
        }
    finally:
        if llm is not None:
            llm.exit()


def format_benchmark_output(rows: list[dict]) -> str:
    if len(rows) == 1:
        row = dict(rows[0])
        row.pop("run", None)
        row.pop("git_commit", None)
        return markdown_table(row)

    summary_keys = [
        "elapsed_s",
        "ttft_s",
        "prefill_time_s",
        "decode_time_s",
        "decode_tokens_per_s",
        "itl_ms_avg",
        "decode_step_ms_p50",
        "decode_step_ms_p95",
        "peak_gpu_memory_gb",
        "max_used_blocks",
        "max_block_utilization",
    ]
    run_columns = [
        "run",
        "generated_tokens",
        "elapsed_s",
        "ttft_s",
        "prefill_time_s",
        "decode_time_s",
        "decode_tokens_per_s",
        "itl_ms_avg",
        "decode_step_ms_p50",
        "decode_step_ms_p95",
        "peak_gpu_memory_gb",
        "num_kvcache_blocks",
        "used_blocks",
        "free_blocks",
        "max_used_blocks",
        "max_block_utilization",
        "prefix_cache_hit_rate",
    ]
    config_keys = [
        "model",
        "gpu",
        "prompt_len",
        "num_prompts",
        "max_tokens",
        "model_dtype",
        "kv_cache_dtype",
        "linear_backend",
        "attn_backend",
        "kvcache_block_size",
        "triton_paged_decode_auto_threshold",
        "norm_backend",
        "activation_backend",
        "rope_backend",
    ]
    config_rows = [{"Metric": key, "Value": rows[0].get(key, "")} for key in config_keys]
    return "\n\n".join([
        "## E2E Config",
        markdown_rows(config_rows, ["Metric", "Value"]),
        "## Per-Run Results",
        markdown_rows(rows, run_columns),
        "## Aggregate Results",
        markdown_rows(aggregate_rows(rows, summary_keys), ["metric", "mean", "p50", "p95"]),
    ])


def main():
    parser = argparse.ArgumentParser(description="End-to-end nano-vLLM generation benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", dest="max_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--block-size", type=int, default=256, choices=[16, 32, 64, 128, 256])
    parser.add_argument(
        "--auto-threshold",
        type=int,
        default=1024,
        help="Context-length threshold for triton_paged_decode_auto.",
    )
    parser.add_argument("--output", type=str, default=None, help="Deprecated alias for --save-md.")
    parser.add_argument("--save-md", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument(
        "--attn-backend",
        default="flash_attn",
        choices=[
            "flash_attn",
            "torch_paged",
            "triton_paged_decode",
            "triton_paged_decode_v2",
            "triton_paged_decode_auto",
        ],
        help="Runtime decode attention backend. Custom paged backends currently require --enforce-eager.",
    )
    args = parser.parse_args()

    assert args.repeat >= 1
    assert args.warmup >= 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_e2e.py")

    from nanovllm import LLM, SamplingParams

    for warmup_idx in range(args.warmup):
        print(f"Warmup {warmup_idx + 1}/{args.warmup}...")
        run_once(args, warmup_idx + 1, LLM, SamplingParams)

    rows = []
    for run_idx in range(args.repeat):
        print(f"Run {run_idx + 1}/{args.repeat}...")
        rows.append(run_once(args, run_idx + 1, LLM, SamplingParams))

    text = format_benchmark_output(rows)
    print(text)
    md_output = args.save_md or args.output
    if md_output:
        output_path = Path(md_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved Markdown summary to {output_path}")
    if args.save_json:
        output_path = Path(args.save_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({
            "benchmark": "bench_e2e",
            "runs": rows,
            "aggregate": aggregate_rows(rows, [
                "elapsed_s",
                "ttft_s",
                "prefill_time_s",
                "decode_time_s",
                "decode_tokens_per_s",
                "itl_ms_avg",
                "decode_step_ms_p50",
                "decode_step_ms_p95",
                "peak_gpu_memory_gb",
                "max_used_blocks",
                "max_block_utilization",
            ]),
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
