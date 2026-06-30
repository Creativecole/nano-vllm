import argparse
from pathlib import Path
import re
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


def categorize_component(name: str) -> str:
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
    if any(token in lowered for token in ("cudalaunch", "culaunch", "cudafunc", "cudadevice")):
        return "Kernel launch/runtime"
    return "Other"


def categorize_cuda_kernel(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("flash_fwd", "flash_attn", "flash::", "splitkv", "_flash_attn")):
        return "Attention"
    if any(token in lowered for token in ("cutlass", "cublas", "gemm", "wmma", "mma")):
        return "Linear/GEMM"
    if "silu_and_mul" in lowered:
        return "Activation"
    if "rms_norm" in lowered or "layer_norm" in lowered:
        return "Normalization"
    if "rotary_embedding" in lowered or "rope" in lowered:
        return "RoPE"
    if "store_kvcache" in lowered or "kvcache" in lowered:
        return "KV cache store"
    if any(token in lowered for token in ("softmax", "argmax", "sample", "sampler")):
        return "Sampling"
    if any(token in lowered for token in ("cudalaunch", "culaunch", "cudafunc", "cudadevice")):
        return "Kernel launch/runtime"
    return "Other"


def is_profiler_wrapper(name: str) -> bool:
    return "nano_vllm_engine_step" in name.lower()


def is_operator_event(name: str) -> bool:
    lowered = name.lower()
    if is_profiler_wrapper(name):
        return False
    if name.startswith("aten::"):
        return True
    if name in {"FlashAttnVarlenFunc"} or lowered.startswith("flash_attn::"):
        return True
    return False


def is_cuda_kernel_event(name: str, self_cuda_us: float) -> bool:
    if self_cuda_us <= 0 or is_operator_event(name) or is_profiler_wrapper(name):
        return False
    lowered = name.lower()
    if lowered.startswith("flash_attn::"):
        return False
    kernel_markers = (
        "kernel",
        "cutlass",
        "cublas",
        "flash_fwd",
        "splitkv",
        "silu_and_mul",
        "rms_norm",
        "rotary_embedding",
        "store_kvcache",
        "cudalaunch",
        "culaunch",
        "cudafunc",
        "cudadevice",
    )
    return name.startswith("void ") or any(marker in lowered for marker in kernel_markers)


def profiler_time_us(event, names: tuple[str, ...]) -> float:
    values = []
    for name in names:
        value = getattr(event, name, None)
        if isinstance(value, (int, float)):
            values.append(float(value))
            if value > 0:
                return float(value)
    return values[0] if values else 0.0


def profiler_self_cuda_us(event) -> float:
    # PyTorch versions differ here: recent profiler events use
    # self_device_time_total/device_time_total, while older versions expose
    # self_cuda_time_total/cuda_time_total. Prefer self time to avoid double
    # counting nested aten::linear -> aten::matmul -> aten::mm stacks.
    return profiler_time_us(event, (
        "self_device_time_total",
        "self_cuda_time_total",
        "device_time_total",
        "cuda_time_total",
    ))


def profiler_self_cpu_us(event) -> float:
    return profiler_time_us(event, (
        "self_cpu_time_total",
        "cpu_time_total",
    ))


def profiler_total_cuda_us(event) -> float:
    return profiler_time_us(event, (
        "device_time_total",
        "cuda_time_total",
        "self_device_time_total",
        "self_cuda_time_total",
    ))


def profiler_total_cpu_us(event) -> float:
    return profiler_time_us(event, (
        "cpu_time_total",
        "self_cpu_time_total",
    ))


def summarize_operator_attribution(prof) -> list[dict]:
    categories = {}
    for event in prof.key_averages():
        if not is_operator_event(event.key):
            continue
        category = categorize_component(event.key)
        entry = categories.setdefault(category, {
            "category": category,
            "cuda_total_ms": 0.0,
            "cpu_total_ms": 0.0,
            "calls": 0,
        })
        entry["cuda_total_ms"] += profiler_total_cuda_us(event) / 1000.0
        entry["cpu_total_ms"] += profiler_total_cpu_us(event) / 1000.0
        entry["calls"] += getattr(event, "count", 0)
    return sorted(categories.values(), key=lambda row: row["cuda_total_ms"], reverse=True)


def summarize_cuda_kernel_categories(prof) -> list[dict]:
    categories = {}
    for event in prof.key_averages():
        self_cuda_us = profiler_self_cuda_us(event)
        if not is_cuda_kernel_event(event.key, self_cuda_us):
            continue
        category = categorize_cuda_kernel(event.key)
        entry = categories.setdefault(category, {
            "category": category,
            "self_cuda_time_ms": 0.0,
            "self_cpu_time_ms": 0.0,
            "calls": 0,
        })
        entry["self_cuda_time_ms"] += self_cuda_us / 1000.0
        entry["self_cpu_time_ms"] += profiler_self_cpu_us(event) / 1000.0
        entry["calls"] += getattr(event, "count", 0)
    return sorted(categories.values(), key=lambda row: row["self_cuda_time_ms"], reverse=True)


def parse_time_to_ms(value: str, unit: str) -> float:
    number = float(value)
    unit = unit.lower()
    if unit == "s":
        return number * 1000.0
    if unit == "ms":
        return number
    if unit == "us":
        return number / 1000.0
    return number


def extract_self_cuda_total_ms(op_table: str) -> float | None:
    match = re.search(r"Self CUDA time total:\s*([0-9.]+)\s*([a-zA-Z]+)", op_table)
    if not match:
        return None
    return parse_time_to_ms(match.group(1), match.group(2))


def kernel_self_time_warning(kernel_rows: list[dict], self_cuda_total_ms: float | None) -> str:
    kernel_total = sum(row["self_cuda_time_ms"] for row in kernel_rows)
    if self_cuda_total_ms is None:
        return (
            f"Kernel category self-time sum: {kernel_total:.4f} ms. "
            "Profiler self CUDA total was not found in the PyTorch table."
        )
    lower = self_cuda_total_ms * 0.90
    upper = self_cuda_total_ms * 1.10
    if kernel_total < lower or kernel_total > upper:
        return (
            f"Warning: kernel category sum differs from profiler self CUDA total by more than 10%; "
            f"kernel category self-time sum: {kernel_total:.4f} ms. "
            f"Profiler self CUDA total: {self_cuda_total_ms:.4f} ms. "
            "Inspect unmatched profiler events."
        )
    return (
        f"Kernel category self-time sum: {kernel_total:.4f} ms. "
        f"Profiler self CUDA total: {self_cuda_total_ms:.4f} ms."
    )


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
    operator_rows = summarize_operator_attribution(prof)
    kernel_rows = summarize_cuda_kernel_categories(prof)
    self_cuda_total_ms = extract_self_cuda_total_ms(op_table)
    return {
        "summary": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "attn_backend": args.attn_backend,
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
        "operator_rows": operator_rows,
        "kernel_rows": kernel_rows,
        "kernel_warning": kernel_self_time_warning(kernel_rows, self_cuda_total_ms),
    }


def main():
    parser = argparse.ArgumentParser(description="Capture a PyTorch profiler trace for nano-vLLM e2e generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--attn-backend",
        default="flash_attn",
        choices=["flash_attn", "torch_paged", "triton_paged_decode"],
        help="Runtime attention backend. Custom paged backends currently require --enforce-eager.",
    )
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
            attn_backend=args.attn_backend,
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
                attn_backend=args.attn_backend,
            )

        result = run_profile(llm, prompts, sampling_params, args)
    finally:
        if llm is not None:
            llm.exit()

    text = "\n\n".join([
        "# nano-vLLM E2E PyTorch Profiler",
        markdown_table(result["summary"]),
        "## Operator-Level CUDA Attribution",
        markdown_rows(result["operator_rows"], ["category", "cuda_total_ms", "cpu_total_ms", "calls"]),
        "Operator attribution is useful for understanding which model components cause CUDA work. "
        "It may include child CUDA kernels, so do not sum it as wall-clock time.",
        "## CUDA Kernel Self-Time Categories",
        markdown_rows(result["kernel_rows"], ["category", "self_cuda_time_ms", "self_cpu_time_ms", "calls"]),
        result["kernel_warning"],
        "Kernel self-time categories are better for deciding low-level optimization targets. "
        "Profiler overhead and CUDA asynchronous execution mean these numbers should explain bottleneck shape, "
        "not replace wall-clock E2E latency.",
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
