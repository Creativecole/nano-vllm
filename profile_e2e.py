import argparse
import json
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
    if any(token in lowered for token in (
        "flash_fwd",
        "flash_attn",
        "flash::",
        "splitkv",
        "_flash_attn",
        "triton_paged_decode",
        "paged_decode",
    )):
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
    rows = sorted(categories.values(), key=lambda row: row["self_cuda_time_ms"], reverse=True)
    for row in rows:
        row["avg_self_cuda_us"] = row["self_cuda_time_ms"] * 1000.0 / row["calls"] if row["calls"] else 0.0
    return rows


def summarize_index_gather_ops(prof) -> list[dict]:
    rows = []
    for event in prof.key_averages():
        lowered = event.key.lower()
        if not any(token in lowered for token in ("aten::index", "gather", "index_select")):
            continue
        calls = getattr(event, "count", 0)
        self_cuda_ms = profiler_self_cuda_us(event) / 1000.0
        cpu_total_ms = profiler_total_cpu_us(event) / 1000.0
        rows.append({
            "name": event.key,
            "self_cuda_time_ms": self_cuda_ms,
            "cpu_total_ms": cpu_total_ms,
            "calls": calls,
            "avg_self_cuda_us": self_cuda_ms * 1000.0 / calls if calls else 0.0,
        })
    return sorted(rows, key=lambda row: row["self_cuda_time_ms"], reverse=True)[:10]


def summarize_selected_ops(prof, tokens: tuple[str, ...], limit: int = 20) -> list[dict]:
    rows = []
    for event in prof.key_averages():
        lowered = event.key.lower()
        if not any(token in lowered for token in tokens):
            continue
        calls = getattr(event, "count", 0)
        self_cuda_ms = profiler_self_cuda_us(event) / 1000.0
        cpu_total_ms = profiler_total_cpu_us(event) / 1000.0
        rows.append({
            "name": event.key,
            "self_cuda_time_ms": self_cuda_ms,
            "cpu_total_ms": cpu_total_ms,
            "calls": calls,
            "avg_self_cuda_us": self_cuda_ms * 1000.0 / calls if calls else 0.0,
        })
    return sorted(rows, key=lambda row: (row["self_cuda_time_ms"], row["cpu_total_ms"]), reverse=True)[:limit]


def summarize_launch_ops(prof) -> list[dict]:
    return summarize_selected_ops(prof, ("cudalaunchkernel", "culaunchkernelex"), limit=10)


def summarize_allocation_copy_ops(prof) -> list[dict]:
    return summarize_selected_ops(
        prof,
        ("aten::contiguous", "aten::copy_", "aten::empty", "aten::empty_like"),
        limit=20,
    )


def summarize_attention_kernel_events(prof) -> list[dict]:
    rows = []
    for event in prof.key_averages():
        self_cuda_us = profiler_self_cuda_us(event)
        if self_cuda_us <= 0:
            continue
        lowered = event.key.lower()
        if not any(token in lowered for token in (
            "triton_paged_decode",
            "paged_decode",
            "flash_fwd",
            "flash_attn",
            "splitkv",
            "_flash_attn",
        )):
            continue
        if is_operator_event(event.key) or is_profiler_wrapper(event.key):
            continue
        calls = getattr(event, "count", 0)
        self_cuda_ms = self_cuda_us / 1000.0
        rows.append({
            "name": event.key,
            "self_cuda_time_ms": self_cuda_ms,
            "calls": calls,
            "avg_self_cuda_us": self_cuda_ms * 1000.0 / calls if calls else 0.0,
        })
    return sorted(rows, key=lambda row: row["self_cuda_time_ms"], reverse=True)[:10]


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
    prefill_time_s = 0.0
    decode_time_s = 0.0

    torch.cuda.synchronize()
    start = perf_counter()
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        while not llm.is_finished() and profiled_steps < args.profile_steps:
            step_start = perf_counter()
            with record_function("nano_vllm_engine_step"):
                output, num_tokens = llm.step()
            step_elapsed = perf_counter() - step_start
            if num_tokens > 0:
                prefill_steps += 1
                prefill_tokens += num_tokens
                prefill_time_s += step_elapsed
            else:
                decode_steps += 1
                decode_tokens += -num_tokens
                decode_time_s += step_elapsed
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
    index_gather_rows = summarize_index_gather_ops(prof)
    attention_kernel_rows = summarize_attention_kernel_events(prof)
    launch_rows = summarize_launch_ops(prof)
    allocation_copy_rows = summarize_allocation_copy_ops(prof)
    self_cuda_total_ms = extract_self_cuda_total_ms(op_table)
    return {
        "summary": {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(),
            "attn_backend": args.attn_backend,
            "block_size": args.block_size,
            "auto_threshold": args.auto_threshold,
            "prompt_len": args.prompt_len,
            "num_prompts": args.num_prompts,
            "max_tokens": args.max_tokens,
            "profiled_steps": profiled_steps,
            "prefill_steps": prefill_steps,
            "decode_steps": decode_steps,
            "prefill_tokens": prefill_tokens,
            "decode_tokens": decode_tokens,
            "prefill_time_s": prefill_time_s,
            "decode_time_s": decode_time_s,
            "elapsed_s": elapsed,
            "trace_output": str(trace_path),
            "profile_memory": args.profile_memory,
            "record_shapes": args.record_shapes,
            "with_stack": args.with_stack,
        },
        "op_table": op_table,
        "operator_rows": operator_rows,
        "kernel_rows": kernel_rows,
        "index_gather_rows": index_gather_rows,
        "attention_kernel_rows": attention_kernel_rows,
        "launch_rows": launch_rows,
        "allocation_copy_rows": allocation_copy_rows,
        "kernel_warning": kernel_self_time_warning(kernel_rows, self_cuda_total_ms),
        "profiler_self_cuda_total_ms": self_cuda_total_ms,
    }


def sum_rows(rows: list[dict], key: str) -> float:
    return sum(float(row.get(key, 0.0)) for row in rows)


def first_matching_row(rows: list[dict], tokens: tuple[str, ...]) -> dict:
    for row in rows:
        name = str(row.get("name", "")).lower()
        if any(token in name for token in tokens):
            return row
    return {}


def category_row(rows: list[dict], category: str) -> dict:
    for row in rows:
        if row.get("category") == category:
            return row
    return {}


def profile_metrics_for_diff(result: dict) -> dict:
    attention_category = category_row(result["kernel_rows"], "Attention")
    launch_rows = result["launch_rows"]
    index_row = first_matching_row(result["index_gather_rows"], ("aten::index",))
    gather_row = first_matching_row(result["index_gather_rows"], ("vectorized_gather_kernel",))
    contiguous_row = first_matching_row(result["allocation_copy_rows"], ("aten::contiguous",))
    copy_row = first_matching_row(result["allocation_copy_rows"], ("aten::copy_",))
    empty_row = first_matching_row(result["allocation_copy_rows"], ("aten::empty",))
    empty_like_row = first_matching_row(result["allocation_copy_rows"], ("aten::empty_like",))
    attention_kernel_self_cuda_ms = sum_rows(result["attention_kernel_rows"], "self_cuda_time_ms")
    attention_kernel_calls = sum(int(row.get("calls", 0)) for row in result["attention_kernel_rows"])
    allocation_rows = result["allocation_copy_rows"]
    allocation_calls = sum(
        int(row.get("calls", 0))
        for row in allocation_rows
        if "aten::empty" in str(row.get("name", "")).lower()
    )
    return {
        "backend": result["summary"]["attn_backend"],
        "elapsed_s": result["summary"]["elapsed_s"],
        "decode_time_s": result["summary"]["decode_time_s"],
        "decode_tokens": result["summary"]["decode_tokens"],
        "profiler_self_cuda_total_ms": result["profiler_self_cuda_total_ms"] or 0.0,
        "cuda_launch_calls": sum(int(row.get("calls", 0)) for row in launch_rows),
        "cuda_launch_cpu_total_ms": sum_rows(launch_rows, "cpu_total_ms"),
        "cuda_launch_self_cuda_ms": sum_rows(launch_rows, "self_cuda_time_ms"),
        "attention_category_self_cuda_ms": float(attention_category.get("self_cuda_time_ms", 0.0)),
        "attention_category_calls": int(attention_category.get("calls", 0)),
        "attention_kernel_self_cuda_ms": attention_kernel_self_cuda_ms,
        "attention_kernel_calls": attention_kernel_calls,
        "attention_kernel_avg_us": attention_kernel_self_cuda_ms * 1000.0 / attention_kernel_calls if attention_kernel_calls else 0.0,
        "aten_index_cpu_total_ms": float(index_row.get("cpu_total_ms", 0.0)),
        "aten_index_self_cuda_ms": float(index_row.get("self_cuda_time_ms", 0.0)),
        "vectorized_gather_self_cuda_ms": float(gather_row.get("self_cuda_time_ms", 0.0)),
        "aten_contiguous_cpu_total_ms": float(contiguous_row.get("cpu_total_ms", 0.0)),
        "aten_copy_cpu_total_ms": float(copy_row.get("cpu_total_ms", 0.0)),
        "aten_copy_self_cuda_ms": float(copy_row.get("self_cuda_time_ms", 0.0)),
        "aten_empty_cpu_total_ms": float(empty_row.get("cpu_total_ms", 0.0)),
        "aten_empty_like_cpu_total_ms": float(empty_like_row.get("cpu_total_ms", 0.0)),
        "total_tensor_allocation_calls": allocation_calls,
    }


def diff_rows(left: dict, right: dict) -> list[dict]:
    keys = [
        "elapsed_s",
        "decode_time_s",
        "profiler_self_cuda_total_ms",
        "cuda_launch_calls",
        "cuda_launch_cpu_total_ms",
        "attention_category_self_cuda_ms",
        "attention_category_calls",
        "attention_kernel_self_cuda_ms",
        "attention_kernel_calls",
        "attention_kernel_avg_us",
        "aten_index_cpu_total_ms",
        "aten_index_self_cuda_ms",
        "vectorized_gather_self_cuda_ms",
        "aten_contiguous_cpu_total_ms",
        "aten_copy_cpu_total_ms",
        "aten_copy_self_cuda_ms",
        "aten_empty_cpu_total_ms",
        "aten_empty_like_cpu_total_ms",
        "total_tensor_allocation_calls",
    ]
    rows = []
    for key in keys:
        v1 = left.get(key, 0)
        v2 = right.get(key, 0)
        delta = v2 - v1
        ratio = (v2 / v1) if isinstance(v1, (int, float)) and v1 else 0.0
        rows.append({
            "metric": key,
            left["backend"]: v1,
            right["backend"]: v2,
            "delta_v2_minus_v1": delta,
            "v2_over_v1": ratio,
        })
    return rows


def write_profile_diff(results: list[dict], args) -> None:
    if len(results) != 2:
        raise ValueError("profile diff currently expects exactly two backends")
    left = profile_metrics_for_diff(results[0])
    right = profile_metrics_for_diff(results[1])
    rows = diff_rows(left, right)
    text = "\n\n".join([
        "# nano-vLLM v1/v2 Profiler Diff",
        markdown_rows(rows, ["metric", left["backend"], right["backend"], "delta_v2_minus_v1", "v2_over_v1"]),
        "Operator attribution and kernel self-time answer different questions; use this diff to locate "
        "where a kernel-level change does or does not survive the full decode runtime.",
    ])
    print(text)
    if args.diff_summary_output:
        path = Path(args.diff_summary_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"\nSaved profiler diff summary to {path}")
    if args.diff_json_output:
        path = Path(args.diff_json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "benchmark": "profile_v1_v2_diff",
            "backends": [left["backend"], right["backend"]],
            "metrics": rows,
            "raw": [left, right],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved profiler diff JSON to {path}")


def main():
    parser = argparse.ArgumentParser(description="Capture a PyTorch profiler trace for nano-vLLM e2e generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--block-size", type=int, default=256, choices=[16, 32, 64, 128, 256])
    parser.add_argument(
        "--auto-threshold",
        type=int,
        default=1024,
        help="Context-length threshold for triton_paged_decode_auto.",
    )
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
        help="Runtime attention backend. Custom paged backends currently require --enforce-eager.",
    )
    parser.add_argument(
        "--compare-backends",
        default=None,
        help="Comma-separated two-backend profiler diff, e.g. triton_paged_decode,triton_paged_decode_v2.",
    )
    parser.add_argument("--diff-summary-output", default=None)
    parser.add_argument("--diff-json-output", default=None)
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

    def run_backend_profile(backend: str, trace_output: str):
        profile_args = argparse.Namespace(**vars(args))
        profile_args.attn_backend = backend
        profile_args.trace_output = trace_output
        llm = None
        try:
            llm = LLM(
                profile_args.model,
                max_model_len=profile_args.prompt_len + profile_args.max_tokens,
                max_num_seqs=profile_args.num_prompts,
                enforce_eager=profile_args.enforce_eager,
                attn_backend=profile_args.attn_backend,
                kvcache_block_size=profile_args.block_size,
                triton_paged_decode_auto_threshold=profile_args.auto_threshold,
            )
            prompt = make_token_prompt(llm, profile_args.prompt_len)
            prompts = [prompt[:] for _ in range(profile_args.num_prompts)]
            sampling_params = SamplingParams(temperature=profile_args.temperature, max_tokens=profile_args.max_tokens)

            if profile_args.warmup_steps:
                run_warmup(llm, [prompt[:] for prompt in prompts], sampling_params, profile_args.warmup_steps)
                llm.exit()
                llm = LLM(
                    profile_args.model,
                    max_model_len=profile_args.prompt_len + profile_args.max_tokens,
                    max_num_seqs=profile_args.num_prompts,
                    enforce_eager=profile_args.enforce_eager,
                    attn_backend=profile_args.attn_backend,
                    kvcache_block_size=profile_args.block_size,
                    triton_paged_decode_auto_threshold=profile_args.auto_threshold,
                )
            return run_profile(llm, prompts, sampling_params, profile_args)
        finally:
            if llm is not None:
                llm.exit()

    if args.compare_backends:
        backends = [backend.strip() for backend in args.compare_backends.split(",") if backend.strip()]
        if len(backends) != 2:
            raise SystemExit("--compare-backends expects exactly two comma-separated backends")
        stem = Path(args.trace_output)
        results = []
        for backend in backends:
            trace_output = str(stem.with_name(f"{stem.stem}_{backend}{stem.suffix or '.json'}"))
            print(f"Profiling backend={backend} trace={trace_output}", flush=True)
            results.append(run_backend_profile(backend, trace_output))
        write_profile_diff(results, args)
        return

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

        if args.warmup_steps:
            run_warmup(llm, [prompt[:] for prompt in prompts], sampling_params, args.warmup_steps)
            llm.exit()
            llm = LLM(
                args.model,
                max_model_len=args.prompt_len + args.max_tokens,
                max_num_seqs=args.num_prompts,
                enforce_eager=args.enforce_eager,
                attn_backend=args.attn_backend,
                kvcache_block_size=args.block_size,
                triton_paged_decode_auto_threshold=args.auto_threshold,
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
        markdown_rows(result["kernel_rows"], [
            "category",
            "self_cuda_time_ms",
            "self_cpu_time_ms",
            "calls",
            "avg_self_cuda_us",
        ]),
        result["kernel_warning"],
        "Kernel self-time categories are better for deciding low-level optimization targets. "
        "Profiler overhead and CUDA asynchronous execution mean these numbers should explain bottleneck shape, "
        "not replace wall-clock E2E latency.",
        "## Attention Kernel Events",
        markdown_rows(result["attention_kernel_rows"], ["name", "self_cuda_time_ms", "calls", "avg_self_cuda_us"]),
        "## Index / Gather Ops",
        markdown_rows(result["index_gather_rows"], ["name", "self_cuda_time_ms", "cpu_total_ms", "calls", "avg_self_cuda_us"]),
        "## Kernel Launch Ops",
        markdown_rows(result["launch_rows"], ["name", "self_cuda_time_ms", "cpu_total_ms", "calls", "avg_self_cuda_us"]),
        "## Allocation / Copy Ops",
        markdown_rows(result["allocation_copy_rows"], ["name", "self_cuda_time_ms", "cpu_total_ms", "calls", "avg_self_cuda_us"]),
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
