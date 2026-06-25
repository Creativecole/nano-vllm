import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile, record_function


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
    lines = [
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in row.items():
        lines.append(f"| {key} | {format_value(value)} |")
    return "\n".join(lines)


def markdown_rows(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def categorize_op(name: str) -> str:
    lowered = name.lower()
    if "nano_vllm_engine_step" in lowered:
        return "Profiler wrapper"
    if any(token in lowered for token in ("cutlass", "cublas", "gemm", "aten::mm", "matmul", "aten::linear")):
        return "Linear/GEMM"
    if "flash" in lowered or "attn" in lowered:
        return "Attention"
    if "rms_norm" in lowered or "layer_norm" in lowered:
        return "Normalization"
    if "silu" in lowered or "gelu" in lowered:
        return "Activation"
    if "rotary" in lowered or "rope" in lowered:
        return "RoPE"
    if "store_kvcache" in lowered or "kvcache" in lowered:
        return "KV cache store"
    if any(token in lowered for token in ("argmax", "softmax", "sample", "sampler")):
        return "Sampling"
    if "cuda" in lowered and "launch" in lowered:
        return "Kernel launch"
    return "Other"


def summarize_profiler_categories(prof) -> list[dict]:
    categories = {}
    for event in prof.key_averages():
        category = categorize_op(event.key)
        if category == "Profiler wrapper":
            continue
        entry = categories.setdefault(category, {
            "category": category,
            "self_cuda_time_ms": 0.0,
            "self_cpu_time_ms": 0.0,
            "calls": 0,
        })
        entry["self_cuda_time_ms"] += getattr(event, "self_cuda_time_total", getattr(event, "cuda_time_total", 0.0)) / 1000.0
        entry["self_cpu_time_ms"] += getattr(event, "self_cpu_time_total", getattr(event, "cpu_time_total", 0.0)) / 1000.0
        entry["calls"] += getattr(event, "count", 0)
    return sorted(categories.values(), key=lambda row: row["self_cuda_time_ms"], reverse=True)


def run_warmup(llm, prompts: list[list[int]], sampling_params, steps: int):
    if steps <= 0:
        return
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)
    for _ in range(steps):
        if llm.is_finished():
            break
        llm.step()
    torch.cuda.synchronize()


def run_profile(llm, prompts: list[list[int]], sampling_params, args):
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    profiled_steps = 0
    prefill_steps = 0
    decode_steps = 0
    prefill_tokens = 0
    decode_tokens = 0

    torch.cuda.synchronize()
    start = perf_counter()
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        while not llm.is_finished() and profiled_steps < args.profile_steps:
            with record_function("nano_vllm_engine_step"):
                output, num_tokens = llm.step()
            if num_tokens > 0:
                prefill_steps += 1
                prefill_tokens += num_tokens
            else:
                decode_steps += 1
                decode_tokens += -num_tokens
            profiled_steps += 1
            prof.step()
    torch.cuda.synchronize()
    elapsed = perf_counter() - start

    trace_path = Path(args.trace_output)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))

    sort_by = "cuda_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
    op_table = prof.key_averages().table(sort_by=sort_by, row_limit=args.row_limit)
    category_rows = summarize_profiler_categories(prof)
    return {
        "summary": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "prompt_len": args.prompt_len,
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "profiled_steps": profiled_steps,
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "elapsed_s": elapsed,
            "trace_output": str(trace_path),
            "profile_memory": args.profile_memory,
            "record_shapes": args.record_shapes,
            "with_stack": args.with_stack,
        },
        "op_table": op_table,
        "category_rows": category_rows,
    }


def main():
    parser = argparse.ArgumentParser(description="Capture a PyTorch profiler trace for nano-vLLM e2e generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--profile-steps", type=int, default=64)
    parser.add_argument("--row-limit", type=int, default=30)
    parser.add_argument("--trace-output", default="profile_e2e_trace.json")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()

    assert args.profile_steps >= 1
    assert args.warmup_steps >= 0

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for profile_e2e.py")

    from nanovllm import LLM, SamplingParams

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

        if args.warmup_steps:
            run_warmup(llm, [prompt[:] for prompt in prompts], sampling_params, args.warmup_steps)
            llm.exit()
            llm = LLM(
                args.model,
                max_model_len=args.prompt_len + args.max_tokens,
                max_num_seqs=args.num_prompts,
                enforce_eager=args.enforce_eager,
            )

        result = run_profile(llm, prompts, sampling_params, args)
    finally:
        if llm is not None:
            llm.exit()

    text = "\n\n".join([
        "# nano-vLLM E2E PyTorch Profiler",
        markdown_table(result["summary"]),
        "## Bottleneck Categories",
        markdown_rows(result["category_rows"], ["category", "self_cuda_time_ms", "self_cpu_time_ms", "calls"]),
        "## Top Ops",
        "```text\n" + result["op_table"] + "\n```",
    ])
    print(text)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved profiler summary to {summary_path}")


if __name__ == "__main__":
    main()
