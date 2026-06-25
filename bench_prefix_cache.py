import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoTokenizer

from bench_e2e import markdown_rows, percentile


def token_from_text(tokenizer, text: str, fallback: int) -> int:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    token_id = token_ids[0] if token_ids else fallback
    eos = tokenizer.eos_token_id
    if token_id == eos:
        return fallback
    return token_id


def repeated_prompt(token_id: int, prompt_len: int) -> list[int]:
    return [token_id] * prompt_len


def mixed_prompt(prefix_token: int, suffix_token: int, prompt_len: int, shared_prefix_len: int) -> list[int]:
    shared_prefix_len = min(prompt_len, shared_prefix_len)
    return [prefix_token] * shared_prefix_len + [suffix_token] * (prompt_len - shared_prefix_len)


def run_generation(llm, prompts: list[list[int]], sampling_params):
    outputs = {}
    prefill_time_s = 0.0
    decode_time_s = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    decode_latencies = []
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
            decode_latencies.append(step_elapsed)
            if first_decode_end is None:
                first_decode_end = perf_counter()

        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = sum(len(token_ids) for token_ids in outputs.values())
    return {
        "elapsed_s": elapsed,
        "ttft_s": (first_decode_end - start) if first_decode_end is not None else 0.0,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "prefill_tokens": prefill_tokens,
        "decode_tokens": decode_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens_per_s": decode_tokens / decode_time_s if decode_time_s else 0.0,
        "itl_ms_avg": decode_time_s / decode_tokens * 1000.0 if decode_tokens else 0.0,
        "decode_step_ms_p50": percentile(decode_latencies, 50) * 1000.0,
        "decode_step_ms_p95": percentile(decode_latencies, 95) * 1000.0,
    }


def metric_delta(after: dict, before: dict, key: str):
    after_value = after.get(key, 0)
    before_value = before.get(key, 0)
    if isinstance(after_value, (int, float)) and isinstance(before_value, (int, float)):
        return after_value - before_value
    return after_value


def build_workloads(tokenizer, args) -> list[dict]:
    block_size = 256
    vocab_size = len(tokenizer)
    base = token_from_text(tokenizer, " benchmark", 1)
    shared = token_from_text(tokenizer, " shared", base)
    fewshot = token_from_text(tokenizer, " example", base)
    system_shared_len = max(block_size, args.prompt_len // 2)
    fewshot_shared_len = max(block_size, (args.prompt_len * 3) // 4)
    return [
        {
            "name": "no_shared_prefix",
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "prime": [],
            "prompts": [repeated_prompt((base + i + 1) % vocab_size, args.prompt_len) for i in range(args.num_prompts)],
            "note": "Each request starts with a different full block, so prefix-cache hits should stay near zero.",
        },
        {
            "name": "shared_system_prompt",
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "prime": [mixed_prompt(shared, base, args.prompt_len, system_shared_len)],
            "prompts": [
                mixed_prompt(shared, (base + i + 17) % vocab_size, args.prompt_len, system_shared_len)
                for i in range(args.num_prompts)
            ],
            "note": "One priming request populates shared system-prompt blocks before measured requests arrive.",
        },
        {
            "name": "shared_few_shot_prefix",
            "prompt_len": args.prompt_len,
            "max_tokens": args.max_tokens,
            "prime": [mixed_prompt(fewshot, base, args.prompt_len, fewshot_shared_len)],
            "prompts": [
                mixed_prompt(fewshot, (base + i + 31) % vocab_size, args.prompt_len, fewshot_shared_len)
                for i in range(args.num_prompts)
            ],
            "note": "A longer shared prefix should increase prefix-cache hit opportunities.",
        },
        {
            "name": "long_prompt_short_decode",
            "prompt_len": args.long_prompt_len,
            "max_tokens": args.short_decode_tokens,
            "prime": [mixed_prompt(shared, base, args.long_prompt_len, max(block_size, args.long_prompt_len // 2))],
            "prompts": [
                mixed_prompt(shared, (base + i + 47) % vocab_size, args.long_prompt_len, max(block_size, args.long_prompt_len // 2))
                for i in range(args.num_prompts)
            ],
            "note": "Stresses prefill and KV block reuse more than decode.",
        },
        {
            "name": "short_prompt_long_decode",
            "prompt_len": args.short_prompt_len,
            "max_tokens": args.long_decode_tokens,
            "prime": [],
            "prompts": [repeated_prompt((base + i + 63) % vocab_size, args.short_prompt_len) for i in range(args.num_prompts)],
            "note": "Stresses decode and ITL; short prompts may not contain enough full blocks for prefix-cache hits.",
        },
    ]


def run_workload(args, workload: dict, LLM, SamplingParams) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    llm = None
    try:
        llm = LLM(
            args.model,
            max_model_len=workload["prompt_len"] + workload["max_tokens"],
            max_num_seqs=args.num_prompts,
            enforce_eager=args.enforce_eager,
        )
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=workload["max_tokens"])

        if workload["prime"]:
            run_generation(llm, workload["prime"], sampling_params)

        before = llm.metrics()
        result = run_generation(llm, workload["prompts"], sampling_params)
        after = llm.metrics()
        prefix_hits = metric_delta(after, before, "prefix_cache_hits")
        prefix_misses = metric_delta(after, before, "prefix_cache_misses")
        prefix_total = prefix_hits + prefix_misses
        return {
            "workload": workload["name"],
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "prompt_len": workload["prompt_len"],
            "num_prompts": args.num_prompts,
            "max_tokens": workload["max_tokens"],
            "generated_tokens": result["generated_tokens"],
            "elapsed_s": result["elapsed_s"],
            "ttft_s": result["ttft_s"],
            "prefill_time_s": result["prefill_time_s"],
            "decode_time_s": result["decode_time_s"],
            "decode_tokens_per_s": result["decode_tokens_per_s"],
            "itl_ms_avg": result["itl_ms_avg"],
            "decode_step_ms_p50": result["decode_step_ms_p50"],
            "decode_step_ms_p95": result["decode_step_ms_p95"],
            "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "num_kvcache_blocks": after["num_kvcache_blocks"],
            "max_used_blocks": after["max_used_blocks"],
            "max_block_utilization": after["max_block_utilization"],
            "prefix_cache_hits": prefix_hits,
            "prefix_cache_misses": prefix_misses,
            "prefix_cache_hit_rate": prefix_hits / prefix_total if prefix_total else 0.0,
            "note": workload["note"],
        }
    finally:
        if llm is not None:
            llm.exit()


def format_report(rows: list[dict]) -> str:
    columns = [
        "workload",
        "prompt_len",
        "max_tokens",
        "ttft_s",
        "decode_tokens_per_s",
        "itl_ms_avg",
        "prefix_cache_hits",
        "prefix_cache_misses",
        "prefix_cache_hit_rate",
        "max_block_utilization",
        "peak_gpu_memory_gb",
    ]
    notes = [{"workload": row["workload"], "note": row["note"]} for row in rows]
    return "\n\n".join([
        "# Prefix Cache Benchmark",
        markdown_rows(rows, columns),
        "## Workload Notes",
        markdown_rows(notes, ["workload", "note"]),
    ])


def main():
    parser = argparse.ArgumentParser(description="Benchmark prefix-cache behavior with several prompt-sharing workloads.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--long-prompt-len", type=int, default=2048)
    parser.add_argument("--short-prompt-len", type=int, default=128)
    parser.add_argument("--short-decode-tokens", type=int, default=32)
    parser.add_argument("--long-decode-tokens", type=int, default=256)
    parser.add_argument("--save-md", type=str, default=None)
    parser.add_argument("--save-json", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Deprecated alias for --save-md.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_prefix_cache.py")

    from nanovllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    workloads = build_workloads(tokenizer, args)

    rows = [run_workload(args, workload, LLM, SamplingParams) for workload in workloads]
    text = format_report(rows)
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
            "benchmark": "bench_prefix_cache",
            "rows": rows,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
