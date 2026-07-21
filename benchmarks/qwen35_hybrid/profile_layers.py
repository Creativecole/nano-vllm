#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    environment_metadata,
    load_model_facts,
    load_nano_text_reference,
    markdown_table,
    require_cuda,
    write_json,
    write_text,
)
from benchmarks.qwen35_hybrid.profile_serving import (  # noqa: E402
    _cpu_time_us,
    _is_cuda_event,
    _time_us,
)
from nanovllm.engine.layer_state import DeltaNetState, PagedKVState  # noqa: E402
from nanovllm.utils.context import reset_context, set_context  # noqa: E402
from nanovllm.utils.profiler import (  # noqa: E402
    configure_profile_ranges,
    profile_range,
)


DEFAULT_JSON = DEFAULT_RESULTS_DIR / "deltanet_backend_profile.json"
DEFAULT_MD = REPO_ROOT / "docs/qwen35_hybrid/06_chunked_recurrence_profile.md"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare one Qwen3.5 Full Attention or DeltaNet serving layer."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--layer-id",
        type=int,
        action="append",
        help="Layer to profile; repeat for multiple layers. Defaults to the first of each type.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Packed prefill request count."
    )
    parser.add_argument(
        "--prompt-len",
        type=int,
        action="append",
        help="Prompt length; repeat for a sweep. Defaults to 128, 512, and 2048.",
    )
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--deltanet-backends",
        default="sequential,chunked",
        help="Comma-separated DeltaNet reference backends to compare.",
    )
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--trace-dir", default=str(DEFAULT_RESULTS_DIR / "traces/layers")
    )
    parser.add_argument("--save-json", default=str(DEFAULT_JSON))
    parser.add_argument("--save-md", default=str(DEFAULT_MD))
    return parser.parse_args()


def resolve_layer_ids(layer_types, requested):
    if requested:
        invalid = [item for item in requested if item < 0 or item >= len(layer_types)]
        if invalid:
            raise ValueError(f"Layer ids out of range: {invalid}")
        return list(dict.fromkeys(requested))
    selected = []
    for layer_type in ("linear_attention", "full_attention"):
        try:
            selected.append(layer_types.index(layer_type))
        except ValueError:
            pass
    if len(selected) != 2:
        raise ValueError("Model must contain both linear_attention and full_attention")
    return selected


def layer_case_key(row):
    return (
        int(row["layer_id"]),
        str(row.get("deltanet_backend", "not_applicable")),
        int(row["batch_size"]),
        int(row["prompt_len"]),
    )


def layer_matrix_spec(args, layer_ids, prompt_lens, deltanet_backends):
    return {
        "layer_ids": layer_ids,
        "batch_size": args.batch_size,
        "prompt_lens": prompt_lens,
        "block_size": args.block_size,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "deltanet_backends": deltanet_backends,
        "deltanet_chunk_size": args.deltanet_chunk_size,
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
    }


def load_layer_checkpoint(args, spec):
    path = Path(args.save_json)
    if args.no_resume or not path.exists():
        return []
    payload = json.loads(path.read_text())
    if (
        payload.get("schema_version") != 2
        or payload.get("environment", {}).get("model") != args.model
        or payload.get("matrix") != spec
    ):
        raise RuntimeError(
            f"Checkpoint {path} belongs to a different layer matrix; "
            "use another output path or --no-resume."
        )
    unique = {layer_case_key(row): row for row in payload.get("profiles", [])}
    print(f"[resume] loaded {len(unique)} completed layer cases", flush=True)
    return list(unique.values())


def make_prefill_context(batch_size, prompt_len, block_size, device):
    total_tokens = batch_size * prompt_len
    cu_seqlens = torch.arange(
        0,
        total_tokens + 1,
        prompt_len,
        dtype=torch.int32,
        device=device,
    )
    blocks_per_sequence = (prompt_len + block_size - 1) // block_size
    slots = []
    for batch_idx in range(batch_size):
        first_slot = batch_idx * blocks_per_sequence * block_size
        slots.extend(range(first_slot, first_slot + prompt_len))
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device=device)
    set_context(
        True,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=prompt_len,
        max_seqlen_k=prompt_len,
        slot_mapping=slot_mapping,
        prefill_seq_lens=(prompt_len,) * batch_size,
    )
    return blocks_per_sequence * batch_size


def make_deltanet_state(layer, batch_size, dtype, device):
    mixer = layer.linear_attn
    return DeltaNetState(
        layer_idx=layer.layer_idx,
        conv_state=torch.zeros(
            batch_size,
            mixer.conv_dim,
            mixer.conv_kernel_size,
            dtype=dtype,
            device=device,
        ),
        recurrent_state=torch.zeros(
            batch_size,
            mixer.num_v_heads,
            mixer.head_k_dim,
            mixer.head_v_dim,
            dtype=torch.float32,
            device=device,
        ),
    )


def bind_attention_cache(layer, num_blocks, block_size, dtype, device):
    mixer = layer.self_attn
    shape = (num_blocks, block_size, mixer.num_kv_heads, mixer.head_dim)
    mixer.bind_paged_state(
        PagedKVState(
            layer_idx=layer.layer_idx,
            k_cache=torch.empty(shape, dtype=dtype, device=device),
            v_cache=torch.empty(shape, dtype=dtype, device=device),
        )
    )


def summarize_layer_profile(prof, repeat, wall_time_s):
    kernels = defaultdict(lambda: {"cuda_time_ms": 0.0, "calls": 0})
    for event in prof.events():
        if not _is_cuda_event(event):
            continue
        name = str(getattr(event, "name", getattr(event, "key", "unknown")))
        duration_us = _time_us(event)
        if duration_us <= 0:
            continue
        kernels[name]["cuda_time_ms"] += duration_us / 1000
        kernels[name]["calls"] += int(getattr(event, "count", 1) or 1)

    top_cuda_kernels = [
        {
            "name": name,
            **values,
            "avg_us_per_call": values["cuda_time_ms"] * 1000 / values["calls"],
        }
        for name, values in kernels.items()
    ]
    top_cuda_kernels.sort(key=lambda item: item["cuda_time_ms"], reverse=True)
    kernel_families = defaultdict(lambda: {"cuda_time_ms": 0.0, "calls": 0})
    for item in top_cuda_kernels:
        lowered = item["name"].lower()
        if "elementwise" in lowered or "pointwise" in lowered:
            family = "elementwise"
        elif "reduce" in lowered:
            family = "reduction"
        elif any(token in lowered for token in ("gemm", "cutlass", "cublas")):
            family = "gemm"
        elif "conv" in lowered or "cudnn" in lowered:
            family = "convolution"
        elif any(token in lowered for token in ("flash", "fmha", "attention")):
            family = "attention"
        else:
            family = "other"
        kernel_families[family]["cuda_time_ms"] += item["cuda_time_ms"]
        kernel_families[family]["calls"] += item["calls"]

    top_operators = []
    cpu_time_ms = 0.0
    ranges = {}
    for event in prof.key_averages():
        name = str(event.key)
        self_cpu_ms = _cpu_time_us(event, self_time=True) / 1000
        self_cuda_ms = _time_us(event, self_time=True) / 1000
        cpu_time_ms += self_cpu_ms
        if name.startswith("qwen35_"):
            ranges[name] = {
                "cuda_total_ms": _time_us(event) / 1000,
                "cpu_total_ms": _cpu_time_us(event) / 1000,
                "calls": int(event.count),
            }
        if self_cpu_ms or self_cuda_ms:
            top_operators.append(
                {
                    "name": name,
                    "self_cuda_time_ms": self_cuda_ms,
                    "self_cpu_time_ms": self_cpu_ms,
                    "calls": int(event.count),
                }
            )
    top_operators.sort(
        key=lambda item: (item["self_cuda_time_ms"], item["self_cpu_time_ms"]),
        reverse=True,
    )
    cuda_time_ms = sum(item["cuda_time_ms"] for item in top_cuda_kernels)
    kernel_count = sum(item["calls"] for item in top_cuda_kernels)
    return {
        "cuda_time": cuda_time_ms,
        "cpu_time": cpu_time_ms,
        "time_unit": "ms",
        "cuda_time_ms": cuda_time_ms,
        "cpu_time_ms": cpu_time_ms,
        "wall_time_ms": wall_time_s * 1000,
        "kernel_count": kernel_count,
        "cuda_time_ms_per_forward": cuda_time_ms / repeat,
        "cpu_time_ms_per_forward": cpu_time_ms / repeat,
        "kernel_count_per_forward": kernel_count / repeat,
        "range_attribution": ranges,
        "kernel_families": dict(kernel_families),
        "top_cuda_kernels": top_cuda_kernels[:20],
        "top_operators": top_operators[:30],
    }


@torch.inference_mode()
def profile_layer(
    model, layer_id, batch_size, prompt_len, deltanet_backend, args
):
    layer = model.model.layers[layer_id]
    layer_type = layer.block_type
    parameter = next(layer.parameters())
    dtype, device = parameter.dtype, parameter.device
    total_tokens = batch_size * prompt_len
    hidden_states = torch.randn(
        total_tokens, model.config.hidden_size, dtype=dtype, device=device
    )
    positions = torch.arange(prompt_len, device=device).repeat(batch_size)
    position_embeddings = model.model.rotary_emb(hidden_states, positions)
    num_blocks = make_prefill_context(
        batch_size, prompt_len, args.block_size, device
    )

    layer_state = None
    if layer_type == "full_attention":
        bind_attention_cache(layer, num_blocks, args.block_size, dtype, device)
        mixer = layer.self_attn
    else:
        layer_state = make_deltanet_state(layer, batch_size, dtype, device)
        mixer = layer.linear_attn
        mixer.deltanet_backend = deltanet_backend
        mixer.deltanet_chunk_size = args.deltanet_chunk_size

    def run_once():
        range_name = (
            "qwen35_full_attention_mixer"
            if layer_type == "full_attention"
            else "qwen35_deltanet_mixer"
        )
        with profile_range(range_name):
            if layer_type == "full_attention":
                return mixer(hidden_states, position_embeddings=position_embeddings)
            return mixer(hidden_states, layer_state=layer_state)

    for _ in range(args.warmup):
        run_once()
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        for _ in range(args.repeat):
            run_once()
    torch.cuda.synchronize()
    wall_time_s = time.perf_counter() - start

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / (
        f"layer{layer_id}_{layer_type}_{deltanet_backend}_"
        f"b{batch_size}_p{prompt_len}.json"
    )
    prof.export_chrome_trace(str(trace_path))
    reset_context()
    summary = summarize_layer_profile(prof, args.repeat, wall_time_s)
    return {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "deltanet_backend": deltanet_backend,
        "deltanet_chunk_size": args.deltanet_chunk_size
        if layer_type == "linear_attention"
        else None,
        "profile_scope": "token_mixer_serving_prefill",
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "repeat": args.repeat,
        "trace": str(trace_path),
        "kernels_per_input_token": (
            summary["kernel_count"] / args.repeat / total_tokens
        ),
        "peak_memory_allocated_gb": torch.cuda.max_memory_allocated() / 2**30,
        "peak_memory_delta_mb": max(
            0, torch.cuda.max_memory_allocated() - baseline_memory
        )
        / 2**20,
        **summary,
    }


def render_markdown(payload):
    rows = payload["profiles"]
    table = markdown_table(
        [
            "layer",
            "type",
            "backend",
            "batch",
            "prompt",
            "CUDA ms/forward",
            "CPU ms/forward",
            "kernels/forward",
            "kernels/input token",
            "peak delta MiB",
        ],
        [
            [
                row["layer_id"],
                row["layer_type"],
                row["deltanet_backend"],
                row["batch_size"],
                row["prompt_len"],
                row["cuda_time_ms_per_forward"],
                row["cpu_time_ms_per_forward"],
                row["kernel_count_per_forward"],
                row["kernels_per_input_token"],
                row["peak_memory_delta_mb"],
            ]
            for row in rows
        ],
    )
    comparisons = payload.get("backend_comparison", [])
    comparison_table = markdown_table(
        [
            "layer",
            "batch",
            "prompt",
            "CUDA speedup",
            "kernel reduction",
            "sequential peak delta MiB",
            "chunked peak delta MiB",
        ],
        [
            [
                row["layer_id"],
                row["batch_size"],
                row["prompt_len"],
                row["cuda_speedup"],
                row["kernel_count_reduction"],
                row["sequential_peak_memory_delta_mb"],
                row["chunked_peak_memory_delta_mb"],
            ]
            for row in comparisons
        ],
    )
    return f"""# Qwen3.5 DeltaNet Kernel Fragmentation

## Code-Level Root Cause

The `sequential` reference executes a Python `for token_idx in range(seq_len)` loop.
The `chunked` reference replaces that sequence-length loop with chunk-level matrix
operations while retaining FP32 recurrent accumulation and BF16 model I/O.

For Qwen3.5-9B, 24 DeltaNet layers at prompt length 2048 produce 49,152 token-layer
iterations. Roughly five elementwise and two reduction launches per iteration predict
about 245,760 elementwise and 98,304 reduction kernels, close to the whole-model
profile observation of 255,009 and 99,792. This identifies the sequential PyTorch
recurrence, rather than GEMM or causal convolution, as the primary launch explosion
candidate.

## Measured Serving-Prefill Layer Comparison

{table}

## Sequential -> Chunked Delta

{comparison_table}

The scope is the real packed serving token mixer: Full Attention uses the existing
FlashAttention varlen path and paged-KV store, while DeltaNet uses its stateful packed
prefill path. Decoder MLP and outer RMSNorm are intentionally excluded.

## Interpretation

- Compare `sequential` and `chunked` kernel count, CUDA time, and temporary-memory
  delta at each prompt length.
- Inspect `qwen35_deltanet_recurrence_sequential`,
  `qwen35_deltanet_recurrence_chunked`, `qwen35_deltanet_conv`, and
  `qwen35_deltanet_output` in each row's `range_attribution` before selecting a target.
- Fusion candidates are the recurrent decay/retrieval/delta/state/output sequence and
  its Q/K normalization. Projection GEMMs and causal convolution should remain library
  calls unless their measured share is material.
- Hugging Face and production vLLM fast paths use chunked/fused Gated DeltaNet and
  optimized causal-convolution/recurrent kernels when available; they avoid issuing a
  Python-controlled chain of elementwise/reduction kernels for every token.

Both paths are non-fused PyTorch references; this experiment changes the execution
model, not the DeltaNet recurrence math.
"""


def compare_deltanet_profiles(rows):
    grouped = defaultdict(dict)
    for row in rows:
        if row["layer_type"] != "linear_attention":
            continue
        key = (row["layer_id"], row["batch_size"], row["prompt_len"])
        grouped[key][row["deltanet_backend"]] = row
    comparisons = []
    for key, backends in sorted(grouped.items()):
        if "sequential" not in backends or "chunked" not in backends:
            continue
        sequential = backends["sequential"]
        chunked = backends["chunked"]
        sequential_cuda = sequential["cuda_time_ms_per_forward"]
        chunked_cuda = chunked["cuda_time_ms_per_forward"]
        sequential_kernels = sequential["kernel_count_per_forward"]
        chunked_kernels = chunked["kernel_count_per_forward"]
        comparisons.append(
            {
                "layer_id": key[0],
                "batch_size": key[1],
                "prompt_len": key[2],
                "cuda_speedup": sequential_cuda / chunked_cuda
                if chunked_cuda
                else None,
                "kernel_count_reduction": 1 - chunked_kernels / sequential_kernels
                if sequential_kernels
                else None,
                "sequential_peak_memory_delta_mb": sequential[
                    "peak_memory_delta_mb"
                ],
                "chunked_peak_memory_delta_mb": chunked["peak_memory_delta_mb"],
            }
        )
    return comparisons


def main():
    args = parse_args()
    require_cuda()
    if (
        args.batch_size <= 0
        or args.warmup < 0
        or args.repeat <= 0
        or args.deltanet_chunk_size <= 0
    ):
        raise ValueError("batch-size/repeat must be positive and warmup non-negative")
    prompt_lens = args.prompt_len or [128, 512, 2048]
    if any(value <= 0 for value in prompt_lens):
        raise ValueError("prompt lengths must be positive")
    deltanet_backends = [
        item.strip() for item in args.deltanet_backends.split(",") if item.strip()
    ]
    invalid_backends = set(deltanet_backends) - {"sequential", "chunked"}
    if not deltanet_backends or invalid_backends:
        raise ValueError(
            f"Unsupported DeltaNet backends: {sorted(invalid_backends)}"
        )

    configure_profile_ranges(torch_ranges=True)
    facts = load_model_facts(args.model)
    layer_ids = resolve_layer_ids(facts["layer_types"], args.layer_id)
    spec = layer_matrix_spec(
        args, layer_ids, prompt_lens, deltanet_backends
    )
    rows = load_layer_checkpoint(args, spec)
    completed = {layer_case_key(row) for row in rows}
    expected = sum(
        len(prompt_lens)
        * (
            len(deltanet_backends)
            if facts["layer_types"][layer_id] == "linear_attention"
            else 1
        )
        for layer_id in layer_ids
    )
    if len(completed) == expected:
        print(f"[resume] all {expected} layer cases are complete", flush=True)
        return
    model, _, load_report = load_nano_text_reference(args.model)
    try:
        for layer_id in layer_ids:
            layer_type = facts["layer_types"][layer_id]
            backends = (
                deltanet_backends
                if layer_type == "linear_attention"
                else ["not_applicable"]
            )
            for deltanet_backend in backends:
                for prompt_len in prompt_lens:
                    key = (
                        layer_id,
                        deltanet_backend,
                        args.batch_size,
                        prompt_len,
                    )
                    if key in completed:
                        print(
                            f"[layer-profile] SKIP layer={layer_id} "
                            f"backend={deltanet_backend} prompt={prompt_len}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[layer-profile] layer={layer_id} type={layer_type} "
                        f"backend={deltanet_backend} batch={args.batch_size} "
                        f"prompt={prompt_len}",
                        flush=True,
                    )
                    row = profile_layer(
                        model,
                        layer_id,
                        args.batch_size,
                        prompt_len,
                        deltanet_backend,
                        args,
                    )
                    rows.append(row)
                    payload = {
                        "schema_version": 2,
                        "environment": environment_metadata(args.model),
                        "model_facts": facts,
                        "matrix": spec,
                        "completed_cases": len(rows),
                        "expected_cases": expected,
                        "weight_load_counts": {
                            "loaded": len(load_report.loaded),
                            "missing": len(load_report.missing),
                            "duplicate": len(load_report.duplicate),
                            "unexpected_text_weights": len(
                                load_report.unexpected_text_weights
                            ),
                            "intentionally_skipped_non_text": len(
                                load_report.intentionally_skipped_non_text
                            ),
                            "tied_aliases": len(load_report.tied_aliases),
                        },
                        "profiles": sorted(rows, key=layer_case_key),
                        "backend_comparison": compare_deltanet_profiles(rows),
                    }
                    write_json(args.save_json, payload)
                    write_text(args.save_md, render_markdown(payload))
                    print(
                        f"[checkpoint] {len(rows)} layer cases -> {args.save_json}",
                        flush=True,
                    )
    finally:
        reset_context()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
