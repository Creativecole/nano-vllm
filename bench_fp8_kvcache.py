import argparse
from time import perf_counter

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.layers.attention import flash_attn_supports_fp8_kvcache


def run_once(args, kv_cache_dtype: str):
    torch.cuda.empty_cache()
    llm = None
    try:
        llm = LLM(
            args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=args.num_prompts,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            kv_cache_dtype=kv_cache_dtype,
            kv_cache_scale=args.kv_cache_scale,
        )
        prompt = " ".join(["benchmark"] * args.prompt_words)
        prompts = [prompt for _ in range(args.num_prompts)]
        sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)

        start = perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = perf_counter() - start
        generated_tokens = sum(len(x["token_ids"]) for x in outputs)
        config = llm.model_runner.config
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        return dict(
            requested_dtype=kv_cache_dtype,
            resolved_dtype=config.resolved_kv_cache_dtype,
            num_kvcache_blocks=config.num_kvcache_blocks,
            generated_tokens=generated_tokens,
            elapsed_s=elapsed,
            tokens_per_s=generated_tokens / elapsed,
            peak_mem_gb=peak_mem,
        )
    finally:
        if llm is not None:
            llm.exit()


def main():
    parser = argparse.ArgumentParser(description="Compare BF16 and FP8 KV cache capacity and generation throughput.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--prompt-words", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-scale", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    rows = [run_once(args, "bf16")]
    if flash_attn_supports_fp8_kvcache():
        rows.append(run_once(args, "fp8_e4m3"))
    else:
        print(
            "\nSkipping fp8_e4m3: this FlashAttention build does not expose "
            "k_descale/v_descale on flash_attn_with_kvcache."
        )
    print("| requested | resolved | kv blocks | tokens/s | elapsed(s) | peak mem(GB) |")
    print("|---|---|---:|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['requested_dtype']} | {row['resolved_dtype']} | "
            f"{row['num_kvcache_blocks']} | {row['tokens_per_s']:.2f} | "
            f"{row['elapsed_s']:.2f} | {row['peak_mem_gb']:.2f} |"
        )
    if len(rows) > 1 and rows[0]["num_kvcache_blocks"]:
        ratio = rows[1]["num_kvcache_blocks"] / rows[0]["num_kvcache_blocks"]
        print(f"\nKV block capacity ratio: {ratio:.2f}x")


if __name__ == "__main__":
    main()
