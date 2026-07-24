#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    deterministic_prompts,
    environment_metadata,
    load_model_facts,
    markdown_table,
    parse_int_list,
    require_cuda,
    safe_error,
    write_json,
    write_text,
)


PROFILE_RANGES = (
    "qwen35_prefill_model",
    "qwen35_decode_model",
    "qwen35_full_attention_mixer",
    "qwen35_deltanet_mixer",
    "qwen35_deltanet_conv",
    "qwen35_deltanet_recurrence",
    "qwen35_deltanet_recurrence_sequential",
    "qwen35_deltanet_recurrence_chunked",
    "qwen35_deltanet_output",
    "qwen35_mlp",
    "qwen35_metadata_prepare",
    "qwen35_metadata_prepare_mixed",
    "qwen35_decode_prepare_fast",
    "qwen35_state_resident_view",
    "qwen35_state_gather",
    "qwen35_state_commit",
)

INTERESTING_OPERATORS = (
    "aten::index",
    "aten::index_select",
    "aten::index_copy_",
    "aten::gather",
    "aten::scatter",
    "aten::copy_",
    "aten::empty",
    "aten::empty_like",
    "aten::empty_strided",
    "aten::contiguous",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile Qwen3.5 hybrid prefill, decode, and continuous batching."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--prompt-lens", default="128,512,2048")
    parser.add_argument("--decode-steps", default="32,128")
    parser.add_argument("--phases", default="prefill,decode,continuous")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--deltanet-backend",
        choices=("sequential", "chunked"),
        default="sequential",
    )
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument(
        "--disable-resident-deltanet-state",
        action="store_true",
        help="Profile the materialized state gather/commit fallback.",
    )
    parser.add_argument(
        "--disable-decode-fast-path",
        action="store_true",
        help="Rebuild decode metadata and resident views on every step for A/B.",
    )
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start a fresh matrix instead of resuming a compatible JSON checkpoint.",
    )
    parser.add_argument(
        "--trace-dir",
        default=str(DEFAULT_RESULTS_DIR / "traces"),
    )
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "profile_analysis.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/05_profile_analysis.md"),
    )
    parser.add_argument(
        "--target-only",
        action="store_true",
        help="Run one NVTX-annotated case for Nsight without PyTorch Profiler.",
    )
    parser.add_argument("--target-phase", choices=("prefill", "decode", "continuous"))
    parser.add_argument("--target-batch", type=int)
    parser.add_argument("--target-prompt", type=int)
    parser.add_argument("--target-decode", type=int)
    return parser.parse_args()


def _time_us(event, self_time=False):
    names = (
        ("self_device_time_total", "self_cuda_time_total")
        if self_time
        else ("device_time_total", "cuda_time_total")
    )
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def _cpu_time_us(event, self_time=False):
    name = "self_cpu_time_total" if self_time else "cpu_time_total"
    return float(getattr(event, name, 0.0) or 0.0)


def _is_cuda_event(event):
    return str(getattr(event, "device_type", "")).lower().endswith("cuda")


def kernel_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("flash_fwd", "flash_attn", "splitkv")):
        return "Full Attention"
    if any(
        token in lowered
        for token in ("cutlass", "cublas", "gemm", "wmma", "tensorop", "mma")
    ):
        return "Linear/GEMM"
    if any(token in lowered for token in ("index_select", "index_copy", "gather", "scatter")):
        return "State gather/scatter"
    if any(token in lowered for token in ("conv", "cudnn")):
        return "Convolution"
    if any(token in lowered for token in ("rms", "norm")):
        return "Normalization"
    if any(token in lowered for token in ("softmax", "argmax", "multinomial")):
        return "Softmax/Sampling"
    if any(token in lowered for token in ("copy", "memcpy", "memset")):
        return "Copy/Memory"
    if any(token in lowered for token in ("elementwise", "vectorized", "reduce", "pointwise")):
        return "Pointwise/Reduction"
    return "Other"


def summarize_profile(prof, wall_time_s: float):
    kernel_categories = defaultdict(lambda: {"self_cuda_time_ms": 0.0, "calls": 0})
    kernels = defaultdict(lambda: {"self_cuda_time_ms": 0.0, "calls": 0})
    for event in prof.events():
        if not _is_cuda_event(event):
            continue
        name = str(getattr(event, "name", getattr(event, "key", "unknown")))
        # record_function ranges also have device attribution, but they are
        # parent annotations rather than leaf CUDA kernels. Counting both the
        # range and its children inflates the kernel total several times over.
        if (
            name in PROFILE_RANGES
            or name.startswith("qwen35_")
            or name.startswith("driver_")
        ):
            continue
        duration_ms = _time_us(event, self_time=False) / 1000
        count = int(getattr(event, "count", 1) or 1)
        category = kernel_category(name)
        kernel_categories[category]["self_cuda_time_ms"] += duration_ms
        kernel_categories[category]["calls"] += count
        kernels[name]["self_cuda_time_ms"] += duration_ms
        kernels[name]["calls"] += count

    ranges = {}
    runtime = defaultdict(lambda: {"self_cpu_time_ms": 0.0, "calls": 0})
    operators = {}
    top_ops = []
    for event in prof.key_averages():
        name = str(event.key)
        if name in PROFILE_RANGES:
            ranges[name] = {
                "cuda_total_ms": _time_us(event) / 1000,
                "cpu_total_ms": _cpu_time_us(event) / 1000,
                "calls": int(event.count),
            }
        lowered = name.lower()
        if any(
            token in lowered
            for token in (
                "cudalaunch",
                "culaunch",
                "cudadevicesynchronize",
                "cudastreamsynchronize",
                "cudaevent",
            )
        ):
            runtime[name]["self_cpu_time_ms"] += (
                _cpu_time_us(event, self_time=True) / 1000
            )
            runtime[name]["calls"] += int(event.count)
        if any(name.startswith(prefix) for prefix in INTERESTING_OPERATORS):
            operators[name] = {
                "self_cuda_time_ms": _time_us(event, self_time=True) / 1000,
                "self_cpu_time_ms": _cpu_time_us(event, self_time=True) / 1000,
                "cpu_total_ms": _cpu_time_us(event) / 1000,
                "calls": int(event.count),
            }
        self_cuda_ms = _time_us(event, self_time=True) / 1000
        self_cpu_ms = _cpu_time_us(event, self_time=True) / 1000
        if self_cuda_ms or self_cpu_ms:
            top_ops.append(
                {
                    "name": name,
                    "self_cuda_time_ms": self_cuda_ms,
                    "self_cpu_time_ms": self_cpu_ms,
                    "calls": int(event.count),
                }
            )

    category_rows = [
        {"category": category, **values}
        for category, values in kernel_categories.items()
    ]
    category_rows.sort(key=lambda row: row["self_cuda_time_ms"], reverse=True)
    kernel_rows = [
        {
            "name": name,
            **values,
            "avg_us_per_call": values["self_cuda_time_ms"] * 1000 / values["calls"],
        }
        for name, values in kernels.items()
        if values["calls"]
    ]
    kernel_rows.sort(key=lambda row: row["self_cuda_time_ms"], reverse=True)
    top_ops.sort(
        key=lambda row: (row["self_cuda_time_ms"], row["self_cpu_time_ms"]),
        reverse=True,
    )
    total_kernel_ms = sum(row["self_cuda_time_ms"] for row in category_rows)
    total_runtime_cpu_ms = sum(
        value["self_cpu_time_ms"] for value in runtime.values()
    )
    for row in category_rows:
        row["cuda_share"] = (
            row["self_cuda_time_ms"] / total_kernel_ms if total_kernel_ms else 0.0
        )
    return {
        "wall_time_s": wall_time_s,
        "kernel_self_cuda_total_ms": total_kernel_ms,
        "runtime_self_cpu_total_ms": total_runtime_cpu_ms,
        "range_attribution": ranges,
        "kernel_categories": category_rows,
        "top_kernels": kernel_rows[:30],
        "runtime": [
            {"name": name, **values} for name, values in runtime.items()
        ],
        "operators": operators,
        "top_ops": top_ops[:50],
    }


@contextmanager
def nvtx_range(name):
    enabled = os.environ.get("NANOVLLM_NVTX") == "1" and torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def add_standard_requests(llm, facts, batch_size, prompt_len, decode_steps):
    from nanovllm import SamplingParams

    prompts = deterministic_prompts(int(facts["vocab_size"]), batch_size, prompt_len)
    params = SamplingParams(
        temperature=0.0, max_tokens=decode_steps, ignore_eos=True
    )
    for prompt in prompts:
        llm.add_request(prompt, params)


def add_continuous_requests(llm, facts, batch_size, prompt_len, decode_steps, start, end):
    from nanovllm import SamplingParams

    for row in range(start, end):
        row_prompt_len = max(1, prompt_len - (row % 3) * max(1, prompt_len // 4))
        row_decode = max(1, decode_steps - (row % 3) * max(1, decode_steps // 4))
        prompt = deterministic_prompts(
            int(facts["vocab_size"]), 1, row_prompt_len, seed=53 + row
        )[0]
        llm.add_request(
            prompt,
            SamplingParams(
                temperature=0.0, max_tokens=row_decode, ignore_eos=True
            ),
        )


def drain(llm):
    while not llm.is_finished():
        llm.step()


def execute_phase(llm, facts, case, phase):
    batch_size, prompt_len, decode_steps = case
    if phase in ("prefill", "decode"):
        add_standard_requests(llm, facts, batch_size, prompt_len, decode_steps)
        if phase == "prefill":
            while llm.scheduler.waiting:
                with nvtx_range("driver_prefill_step"):
                    llm.step()
            return
        while llm.scheduler.waiting:
            llm.step()
        while not llm.is_finished():
            with nvtx_range("driver_decode_step"):
                llm.step()
        return

    initial = max(1, batch_size // 2)
    add_continuous_requests(
        llm, facts, batch_size, prompt_len, decode_steps, 0, initial
    )
    with nvtx_range("driver_continuous_initial_step"):
        llm.step()
    add_continuous_requests(
        llm, facts, batch_size, prompt_len, decode_steps, initial, batch_size
    )
    while not llm.is_finished():
        phase_name = (
            "driver_continuous_prefill_step"
            if llm.scheduler.waiting
            else "driver_continuous_decode_step"
        )
        with nvtx_range(phase_name):
            llm.step()


def run_profile_case(llm, facts, case, phase, trace_path, args):
    for _ in range(args.warmup):
        execute_phase(llm, facts, case, phase)
        drain(llm)
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        execute_phase(llm, facts, case, phase)
    torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - start
    if not llm.is_finished():
        drain(llm)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    return summarize_profile(prof, wall_time_s)


def matrix_spec(args):
    return {
        "batch_sizes": parse_int_list(args.batch_sizes),
        "prompt_lens": parse_int_list(args.prompt_lens),
        "decode_steps": parse_int_list(args.decode_steps),
        "phases": [item.strip() for item in args.phases.split(",") if item.strip()],
        "warmup": args.warmup,
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
        "deltanet_backend": args.deltanet_backend,
        "deltanet_chunk_size": args.deltanet_chunk_size,
        "resident_deltanet_state": (
            not args.disable_resident_deltanet_state
        ),
        "decode_fast_path": not args.disable_decode_fast_path,
    }


def case_key(item):
    return (
        int(item["batch_size"]),
        int(item["prompt_len"]),
        int(item["decode_steps"]),
        str(item["phase"]),
    )


def load_checkpoint(args, spec):
    path = Path(args.save_json)
    if args.no_resume or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot resume invalid checkpoint {path}: {exc}. "
            "Use --no-resume to replace it."
        ) from exc
    if payload.get("schema_version") != 1:
        raise RuntimeError(
            f"Checkpoint {path} has an unsupported schema. Use --no-resume."
        )
    saved_model = payload.get("environment", {}).get("model")
    if saved_model != args.model or payload.get("matrix") != spec:
        raise RuntimeError(
            f"Checkpoint {path} belongs to a different model or matrix. "
            "Use another --save-json path or pass --no-resume."
        )
    rows = payload.get("profiles", [])
    unique = {}
    for row in rows:
        unique[case_key(row)] = row
    print(f"[resume] loaded {len(unique)} completed cases from {path}", flush=True)
    return list(unique.values())


def build_payload(args, facts, spec, rows, failures):
    expected_cases = (
        len(spec["batch_sizes"])
        * len(spec["prompt_lens"])
        * len(spec["decode_steps"])
        * len(spec["phases"])
    )
    return {
        "schema_version": 1,
        "environment": environment_metadata(args.model),
        "model_facts": facts,
        "matrix": spec,
        "completed_cases": len(rows),
        "expected_cases": expected_cases,
        "profiles": sorted(rows, key=case_key),
        "answers": derive_answers(rows),
        "failures": failures,
    }


def save_checkpoint(args, facts, spec, rows, failures):
    payload = build_payload(args, facts, spec, rows, failures)
    write_json(args.save_json, payload)
    write_text(args.save_md, render_markdown(payload))
    print(
        f"[checkpoint] {len(rows)} completed -> {args.save_json}",
        flush=True,
    )


def profile_matrix(args, facts, spec, rows):
    require_cuda()
    os.environ["NANOVLLM_PROFILE_RANGES"] = "1"
    from nanovllm import LLM
    from nanovllm.utils.profiler import configure_profile_ranges

    configure_profile_ranges(torch_ranges=True)

    batch_sizes = spec["batch_sizes"]
    prompt_lens = spec["prompt_lens"]
    decode_steps = spec["decode_steps"]
    phases = spec["phases"]
    invalid = set(phases) - {"prefill", "decode", "continuous"}
    if invalid:
        raise ValueError(f"Unsupported phases: {sorted(invalid)}")
    completed = {case_key(row) for row in rows}
    expected = len(batch_sizes) * len(prompt_lens) * len(decode_steps) * len(phases)
    if len(completed) == expected:
        print(f"[resume] all {expected} cases are already complete", flush=True)
        return rows, []
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=max(batch_sizes),
        hybrid_state_capacity=max(batch_sizes),
        max_model_len=max(prompt_lens) + max(decode_steps) + 1,
        max_num_batched_tokens=max(batch_sizes) * max(prompt_lens),
        resident_deltanet_state=not args.disable_resident_deltanet_state,
        decode_fast_path=not args.disable_decode_fast_path,
        deltanet_backend=args.deltanet_backend,
        deltanet_chunk_size=args.deltanet_chunk_size,
    )
    failures = []
    trace_dir = Path(args.trace_dir)
    try:
        for batch_size in batch_sizes:
            for prompt_len in prompt_lens:
                for decode in decode_steps:
                    case = (batch_size, prompt_len, decode)
                    for phase in phases:
                        label = (
                            f"{args.deltanet_backend}_b{batch_size}_"
                            f"p{prompt_len}_d{decode}_{phase}"
                        )
                        trace_path = trace_dir / f"{label}.json"
                        key = (batch_size, prompt_len, decode, phase)
                        if key in completed:
                            print(f"[profile] SKIP completed {label}", flush=True)
                            continue
                        print(f"[profile] {label}", flush=True)
                        try:
                            summary = run_profile_case(
                                llm, facts, case, phase, trace_path, args
                            )
                            rows.append(
                                {
                                    "batch_size": batch_size,
                                    "prompt_len": prompt_len,
                                    "decode_steps": decode,
                                    "phase": phase,
                                    "deltanet_backend": args.deltanet_backend,
                                    "deltanet_chunk_size": args.deltanet_chunk_size,
                                    "trace": str(trace_path),
                                    **summary,
                                }
                            )
                            completed.add(key)
                            save_checkpoint(args, facts, spec, rows, failures)
                        except Exception as exc:
                            failures.append(
                                {
                                    "batch_size": batch_size,
                                    "prompt_len": prompt_len,
                                    "decode_steps": decode,
                                    "phase": phase,
                                    "error": safe_error(exc),
                                }
                            )
                            print(f"[profile] FAILED {label}: {exc}", flush=True)
                            save_checkpoint(args, facts, spec, rows, failures)
                            return rows, failures
    finally:
        llm.exit()
    return rows, failures


def run_target(args):
    require_cuda()
    os.environ["NANOVLLM_NVTX"] = "1"
    from nanovllm import LLM
    from nanovllm.utils.profiler import configure_profile_ranges

    configure_profile_ranges(nvtx=True)

    required = (
        args.target_phase,
        args.target_batch,
        args.target_prompt,
        args.target_decode,
    )
    if any(value is None for value in required):
        raise ValueError("target mode requires phase, batch, prompt, and decode")
    facts = load_model_facts(args.model)
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=args.target_batch,
        hybrid_state_capacity=args.target_batch,
        max_model_len=args.target_prompt + args.target_decode + 1,
        max_num_batched_tokens=args.target_batch * args.target_prompt,
        resident_deltanet_state=not args.disable_resident_deltanet_state,
        decode_fast_path=not args.disable_decode_fast_path,
        deltanet_backend=args.deltanet_backend,
        deltanet_chunk_size=args.deltanet_chunk_size,
    )
    try:
        case = (args.target_batch, args.target_prompt, args.target_decode)
        for _ in range(args.warmup):
            execute_phase(llm, facts, case, args.target_phase)
            drain(llm)
        with nvtx_range("qwen35_profile_target"):
            execute_phase(llm, facts, case, args.target_phase)
        if args.target_phase != "prefill":
            drain(llm)
        torch.cuda.synchronize()
    finally:
        llm.exit()


def _range_value(row, name):
    return row["range_attribution"].get(name, {}).get("cuda_total_ms", 0.0)


def derive_answers(rows):
    trend_rows = []
    category_totals = defaultdict(float)
    kernel_totals = defaultdict(float)
    phase_totals = defaultdict(
        lambda: {
            "profiles": 0,
            "wall_time_s": 0.0,
            "kernel_self_cuda_ms": 0.0,
            "runtime_self_cpu_ms": 0.0,
            "full_attention_cuda_ms": 0.0,
            "deltanet_cuda_ms": 0.0,
            "mlp_cuda_ms": 0.0,
            "state_gather_cuda_ms": 0.0,
            "state_gather_cpu_ms": 0.0,
            "state_commit_cuda_ms": 0.0,
            "state_commit_cpu_ms": 0.0,
            "state_resident_cpu_ms": 0.0,
            "metadata_prepare_cpu_ms": 0.0,
            "metadata_prepare_calls": 0,
            "decode_fast_prepare_cpu_ms": 0.0,
            "decode_fast_prepare_calls": 0,
            "recurrence_cuda_ms": 0.0,
            "recurrence_calls": 0,
        }
    )
    for row in rows:
        full = _range_value(row, "qwen35_full_attention_mixer")
        delta = _range_value(row, "qwen35_deltanet_mixer")
        mlp = _range_value(row, "qwen35_mlp")
        mixer_total = full + delta + mlp
        categories = {
            item["category"]: item["self_cuda_time_ms"]
            for item in row["kernel_categories"]
        }
        for name, value in categories.items():
            category_totals[name] += value
        for kernel in row["top_kernels"]:
            kernel_totals[kernel["name"]] += kernel["self_cuda_time_ms"]
        ranges = row["range_attribution"]
        recurrence = ranges.get("qwen35_deltanet_recurrence_chunked", {})
        if not recurrence:
            recurrence = ranges.get("qwen35_deltanet_recurrence_sequential", {})
        if not recurrence:
            recurrence = ranges.get("qwen35_deltanet_recurrence", {})
        phase_total = phase_totals[row["phase"]]
        phase_total["profiles"] += 1
        phase_total["wall_time_s"] += row["wall_time_s"]
        phase_total["kernel_self_cuda_ms"] += row["kernel_self_cuda_total_ms"]
        phase_total["runtime_self_cpu_ms"] += row["runtime_self_cpu_total_ms"]
        phase_total["full_attention_cuda_ms"] += full
        phase_total["deltanet_cuda_ms"] += delta
        phase_total["mlp_cuda_ms"] += mlp
        for range_name, prefix in (
            ("qwen35_state_gather", "state_gather"),
            ("qwen35_state_commit", "state_commit"),
        ):
            values = row["range_attribution"].get(range_name, {})
            phase_total[f"{prefix}_cuda_ms"] += values.get("cuda_total_ms", 0.0)
            phase_total[f"{prefix}_cpu_ms"] += values.get("cpu_total_ms", 0.0)
        phase_total["state_resident_cpu_ms"] += ranges.get(
            "qwen35_state_resident_view", {}
        ).get("cpu_total_ms", 0.0)
        for range_name in (
            "qwen35_metadata_prepare",
            "qwen35_metadata_prepare_mixed",
        ):
            prepare = ranges.get(range_name, {})
            phase_total["metadata_prepare_cpu_ms"] += prepare.get(
                "cpu_total_ms",
                0.0,
            )
            phase_total["metadata_prepare_calls"] += prepare.get("calls", 0)
        fast_prepare = ranges.get("qwen35_decode_prepare_fast", {})
        phase_total["decode_fast_prepare_cpu_ms"] += fast_prepare.get(
            "cpu_total_ms",
            0.0,
        )
        phase_total["decode_fast_prepare_calls"] += fast_prepare.get(
            "calls",
            0,
        )
        phase_total["recurrence_cuda_ms"] += recurrence.get("cuda_total_ms", 0.0)
        phase_total["recurrence_calls"] += recurrence.get("calls", 0)
        trend_rows.append(
            {
                "batch_size": row["batch_size"],
                "prompt_len": row["prompt_len"],
                "decode_steps": row["decode_steps"],
                "phase": row["phase"],
                "full_attention_range_cuda_pct": full / mixer_total
                if mixer_total
                else None,
                "deltanet_range_cuda_pct": delta / mixer_total
                if mixer_total
                else None,
                "mlp_range_cuda_pct": mlp / mixer_total if mixer_total else None,
                "linear_gemm_kernel_pct": categories.get("Linear/GEMM", 0.0)
                / row["kernel_self_cuda_total_ms"]
                if row["kernel_self_cuda_total_ms"]
                else None,
                "state_gather_scatter_kernel_pct": categories.get(
                    "State gather/scatter", 0.0
                )
                / row["kernel_self_cuda_total_ms"]
                if row["kernel_self_cuda_total_ms"]
                else None,
                "recurrence_cuda_total_ms": recurrence.get("cuda_total_ms"),
                "recurrence_calls": recurrence.get("calls"),
                "recurrence_avg_us": recurrence.get("cuda_total_ms", 0.0)
                * 1000
                / recurrence.get("calls", 1)
                if recurrence
                else None,
            }
        )
    hottest_category = max(category_totals, key=category_totals.get) if category_totals else None
    hottest_kernels = sorted(
        kernel_totals.items(), key=lambda item: item[1], reverse=True
    )[:2]
    total_component_ms = sum(
        values[name]
        for values in phase_totals.values()
        for name in (
            "full_attention_cuda_ms",
            "deltanet_cuda_ms",
            "mlp_cuda_ms",
        )
    )
    total_kernel_ms = sum(category_totals.values())
    total_recurrence_ms = sum(
        values["recurrence_cuda_ms"] for values in phase_totals.values()
    )
    total_recurrence_calls = sum(
        values["recurrence_calls"] for values in phase_totals.values()
    )
    overall = {
        "full_attention_component_share": (
            sum(v["full_attention_cuda_ms"] for v in phase_totals.values())
            / total_component_ms
            if total_component_ms
            else None
        ),
        "deltanet_component_share": (
            sum(v["deltanet_cuda_ms"] for v in phase_totals.values())
            / total_component_ms
            if total_component_ms
            else None
        ),
        "linear_gemm_kernel_share": (
            category_totals.get("Linear/GEMM", 0.0) / total_kernel_ms
            if total_kernel_ms
            else None
        ),
        "state_gather_scatter_kernel_share": (
            category_totals.get("State gather/scatter", 0.0) / total_kernel_ms
            if total_kernel_ms
            else None
        ),
        "recurrence_calls": total_recurrence_calls,
        "recurrence_avg_us": (
            total_recurrence_ms * 1000 / total_recurrence_calls
            if total_recurrence_calls
            else None
        ),
    }

    def grouped_trend(field, metric):
        grouped = defaultdict(list)
        for item in trend_rows:
            value = item[metric]
            if value is not None:
                grouped[item[field]].append(value)
        return [
            {field: key, metric: sum(values) / len(values)}
            for key, values in sorted(grouped.items())
        ]

    phase_rows = []
    for phase, values in sorted(phase_totals.items()):
        component_total = (
            values["full_attention_cuda_ms"]
            + values["deltanet_cuda_ms"]
            + values["mlp_cuda_ms"]
        )
        phase_rows.append(
            {
                "phase": phase,
                **values,
                "full_attention_component_share": (
                    values["full_attention_cuda_ms"] / component_total
                    if component_total
                    else None
                ),
                "deltanet_component_share": (
                    values["deltanet_cuda_ms"] / component_total
                    if component_total
                    else None
                ),
                "mlp_component_share": (
                    values["mlp_cuda_ms"] / component_total
                    if component_total
                    else None
                ),
                "recurrence_avg_us": (
                    values["recurrence_cuda_ms"] * 1000
                    / values["recurrence_calls"]
                    if values["recurrence_calls"]
                    else None
                ),
            }
        )
    return {
        "trend_rows": trend_rows,
        "phase_summary": phase_rows,
        "overall": overall,
        "batch_full_attention_trend": grouped_trend(
            "batch_size", "full_attention_range_cuda_pct"
        ),
        "context_full_attention_trend": grouped_trend(
            "prompt_len", "full_attention_range_cuda_pct"
        ),
        "aggregate_kernel_categories_ms": dict(category_totals),
        "recommended_single_hotspot": hottest_category,
        "top_two_kernels": [
            {"name": name, "self_cuda_time_ms": value}
            for name, value in hottest_kernels
        ],
    }


def render_markdown(payload):
    answers = payload["answers"]

    def pct(value):
        return None if value is None else value * 100

    def pct_text(value):
        return "N/A" if value is None else f"{value * 100:.2f}%"

    def number_text(value, suffix=""):
        return "N/A" if value is None else f"{value:.3f}{suffix}"

    trends = markdown_table(
        [
            "phase",
            "batch",
            "prompt",
            "decode",
            "Full Attn range %",
            "DeltaNet range %",
            "MLP range %",
            "GEMM kernel %",
            "state G/S kernel %",
            "recurrence us/call",
        ],
        [
            [
                row["phase"],
                row["batch_size"],
                row["prompt_len"],
                row["decode_steps"],
                pct(row["full_attention_range_cuda_pct"]),
                pct(row["deltanet_range_cuda_pct"]),
                pct(row["mlp_range_cuda_pct"]),
                pct(row["linear_gemm_kernel_pct"]),
                pct(row["state_gather_scatter_kernel_pct"]),
                row["recurrence_avg_us"],
            ]
            for row in answers["trend_rows"]
        ],
    )
    categories = markdown_table(
        ["kernel category", "aggregate self CUDA ms"],
        sorted(
            answers["aggregate_kernel_categories_ms"].items(),
            key=lambda item: item[1],
            reverse=True,
        ),
    )
    batch_trend = markdown_table(
        ["batch", "mean Full Attn component %"],
        [
            [row["batch_size"], pct(row["full_attention_range_cuda_pct"])]
            for row in answers["batch_full_attention_trend"]
        ],
    )
    context_trend = markdown_table(
        ["prompt length", "mean Full Attn component %"],
        [
            [row["prompt_len"], pct(row["full_attention_range_cuda_pct"])]
            for row in answers["context_full_attention_trend"]
        ],
    )
    phase_summary = markdown_table(
        [
            "phase",
            "profiles",
            "Full Attn component %",
            "DeltaNet component %",
            "MLP component %",
            "kernel self CUDA ms",
            "CUDA runtime self CPU ms",
            "state gather CUDA ms",
            "state commit CUDA ms",
            "resident state view CPU ms",
            "normal prepare CPU ms/calls",
            "fast prepare CPU ms/calls",
            "recurrence us/call",
        ],
        [
            [
                row["phase"],
                row["profiles"],
                pct(row["full_attention_component_share"]),
                pct(row["deltanet_component_share"]),
                pct(row["mlp_component_share"]),
                row["kernel_self_cuda_ms"],
                row["runtime_self_cpu_ms"],
                row["state_gather_cuda_ms"],
                row["state_commit_cuda_ms"],
                row["state_resident_cpu_ms"],
                (
                    f"{row['metadata_prepare_cpu_ms']:.3f}/"
                    f"{row['metadata_prepare_calls']}"
                ),
                (
                    f"{row['decode_fast_prepare_cpu_ms']:.3f}/"
                    f"{row['decode_fast_prepare_calls']}"
                ),
                row["recurrence_avg_us"],
            ]
            for row in answers["phase_summary"]
        ],
    )
    failures = payload["failures"]
    failure_text = "None." if not failures else "\n".join(f"- {item}" for item in failures)
    return f"""# Qwen3.5 Hybrid Serving Profile Analysis

Completed cases: `{payload.get('completed_cases', len(payload['profiles']))}` / `{payload.get('expected_cases', len(payload['profiles']))}`.

This report separates additive CUDA kernel self-time from high-level operator/range
attribution. Range percentages describe model components and may contain child kernels;
kernel category percentages use leaf CUDA device events after excluding high-level
`qwen35_*` and driver annotations, and are the better low-level target signal.

## Phase Summary

{phase_summary}

Full Attention, DeltaNet, and MLP percentages use their combined annotated CUDA time
as the denominator. They are component attribution, not additive wall-clock shares.

## Attribution By Workload

{trends}

## CUDA Kernel Self-Time

{categories}

## Profiler-Guided Decision

- Current hottest kernel category: `{answers['recommended_single_hotspot']}`
- Top two concrete kernels: `{answers['top_two_kernels']}`

This recommendation is generated from measured profiler data. It is not a preset
assumption that DeltaNet or Full Attention must be optimized next.

## Questions This Run Must Answer

1. **Full Attention share:** `{pct_text(answers['overall']['full_attention_component_share'])}` of aggregate annotated Full Attention + DeltaNet + MLP CUDA time. This is component attribution, not wall-clock share.
2. **DeltaNet share:** `{pct_text(answers['overall']['deltanet_component_share'])}` on the same denominator.
3. **Linear/GEMM dominance:** `{pct_text(answers['overall']['linear_gemm_kernel_share'])}` of additive CUDA leaf-kernel self-time is categorized as Linear/GEMM.
4. **Small recurrent kernels:** DeltaNet recurrence is called `{answers['overall']['recurrence_calls']}` times at `{number_text(answers['overall']['recurrence_avg_us'], ' us/call')}` on average. Confirm launch spacing in Nsight Systems.
5. **State gather/scatter:** `{pct_text(answers['overall']['state_gather_scatter_kernel_share'])}` of additive kernel self-time is directly categorized as gather/scatter; range totals are in the phase table and raw JSON.
6. **Batch scaling:** the measured Full Attention component trend is shown below. Compare fixed-shape rows in the detailed table before drawing a causal conclusion.
7. **Context scaling:** the measured prompt-length trend is shown below; longer prompt primarily affects prefill, while decode also depends on accumulated Full Attention KV length.
8. **Next optimization target:** `{answers['recommended_single_hotspot']}` is the current hottest leaf-kernel category. Confirm `{answers['top_two_kernels']}` with Nsight Compute before changing code.

### Batch Trend

{batch_trend}

### Context Trend

{context_trend}

## Failed Configurations

{failure_text}

## Nsight Follow-up

Use `run_nsight.py` on a representative row after inspecting the PyTorch profile.
Nsight Systems validates launch gaps, synchronization, and overlap. Nsight Compute
should then target only the one or two hottest concrete kernels listed above.
"""


def main():
    args = parse_args()
    if args.deltanet_chunk_size <= 0:
        raise ValueError("--deltanet-chunk-size must be positive")
    if args.target_only:
        run_target(args)
        return
    spec = matrix_spec(args)
    invalid = set(spec["phases"]) - {"prefill", "decode", "continuous"}
    if invalid:
        raise ValueError(f"Unsupported phases: {sorted(invalid)}")
    facts = load_model_facts(args.model)
    rows = load_checkpoint(args, spec)
    rows, failures = profile_matrix(args, facts, spec, rows)
    save_checkpoint(args, facts, spec, rows, failures)
    print(f"Saved {args.save_json}", flush=True)
    print(f"Saved {args.save_md}", flush=True)


if __name__ == "__main__":
    main()
