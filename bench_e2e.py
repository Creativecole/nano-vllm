import argparse
from pathlib import Path
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


def main():
    parser = argparse.ArgumentParser(description="End-to-end nano-vLLM generation benchmark.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_e2e.py")

    from nanovllm import LLM, SamplingParams

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    llm = None
    try:
        llm = LLM(
            args.model,
            max_model_len=args.prompt_len + args.max_tokens,
            max_num_seqs=args.num_prompts,
            enforce_eager=args.enforce_eager,
        )
        prompt = make_token_prompt(llm, args.prompt_len)
        prompts = [prompt[:] for _ in range(args.num_prompts)]
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        torch.cuda.synchronize()
        start = perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = perf_counter() - start

        generated_tokens = sum(len(output["token_ids"]) for output in outputs)
        prompt_tokens = args.prompt_len * args.num_prompts
        total_tokens = prompt_tokens + generated_tokens
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1024**3
        metrics = llm.metrics()
        kv_dtype = metrics["kv_cache_dtype"]
        row = {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "prompt_len": args.prompt_len,
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "elapsed_s": elapsed,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "total_tokens_per_s": total_tokens / elapsed,
            "approx_decode_tokens_per_s": generated_tokens / elapsed,
            "peak_gpu_memory_gb": peak_mem_gb,
            "num_kvcache_blocks": metrics["num_kvcache_blocks"],
            "used_blocks": metrics["used_blocks"],
            "free_blocks": metrics["free_blocks"],
            "block_utilization": metrics["block_utilization"],
            "max_used_blocks": metrics["max_used_blocks"],
            "max_block_utilization": metrics["max_block_utilization"],
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
        }
        text = markdown_table(row)
        print(text)
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(text + "\n", encoding="utf-8")
            print(f"\nSaved Markdown summary to {output_path}")
    finally:
        if llm is not None:
            llm.exit()


if __name__ == "__main__":
    main()
