from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nanovllm.backends import AttentionBackendError, paged_attention_decode
from nanovllm.kernels.attention import (
    build_block_tables,
    pack_dense_kv_to_cache,
    torch_paged_attention_decode,
)
from nanovllm.utils.shapes import load_qwen_attention_shapes


def log(message: str) -> None:
    print(message, flush=True)


def describe_case(args, shapes, batch_size: int, seq_len: int, block_size: int) -> str:
    return (
        f"backend={args.backend}, batch_size={batch_size}, seq_len={seq_len}, "
        f"block_size={block_size}, dtype={args.dtype}"
    )


def verbose_case_details(args, shapes, q=None, k_cache=None, v_cache=None, block_tables=None, context_lens=None) -> None:
    if not getattr(args, "verbose", False):
        return
    log(
        "[verbose] "
        f"num_q_heads={shapes.num_attention_heads}, "
        f"num_kv_heads={shapes.num_key_value_heads}, "
        f"head_dim={shapes.head_dim}, "
        f"gqa_ratio={shapes.gqa_ratio}, "
        f"dtype={args.dtype}"
    )
    if q is not None:
        log(
            "[verbose] "
            f"q={tuple(q.shape)}, k_cache={tuple(k_cache.shape)}, "
            f"v_cache={tuple(v_cache.shape)}, block_tables={tuple(block_tables.shape)}, "
            f"context_lens={tuple(context_lens.shape)}"
        )


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unsupported dtype: {name}") from exc


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def format_value(value):
    if value is None:
        return "skipped"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_No rows generated._"
    columns = list(rows[0].keys())
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def make_decode_inputs(
    batch_size: int,
    seq_len: int,
    block_size: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.randn(batch_size, num_q_heads, head_dim, device=device, dtype=dtype)
    dense_k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    dense_v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    block_tables = build_block_tables(batch_size, seq_len, block_size, device=device)
    context_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    k_cache, v_cache = pack_dense_kv_to_cache(dense_k, dense_v, block_tables, block_size)
    return q, k_cache, v_cache, block_tables, context_lens


def run_case(args, shapes, batch_size: int, seq_len: int, block_size: int) -> dict:
    case = describe_case(args, shapes, batch_size, seq_len, block_size)
    log(f"[case] start {case}")
    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.backend.startswith("triton_paged_decode") and device.type != "cuda":
        raise AttentionBackendError(f"{args.backend} requires CUDA")
    log(f"[case] build inputs {case}")
    q, k_cache, v_cache, block_tables, context_lens = make_decode_inputs(
        batch_size=batch_size,
        seq_len=seq_len,
        block_size=block_size,
        num_q_heads=shapes.num_attention_heads,
        num_kv_heads=shapes.num_key_value_heads,
        head_dim=shapes.head_dim,
        dtype=dtype,
        device=device,
    )
    verbose_case_details(args, shapes, q, k_cache, v_cache, block_tables, context_lens)
    scale = shapes.head_dim ** -0.5
    log(f"[case] correctness {case}")
    reference = torch_paged_attention_decode(q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    if args.backend == "torch_paged":
        output = reference
    else:
        output = paged_attention_decode(args.backend, q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    if device.type == "cuda":
        torch.cuda.synchronize()
    diff = (output.float() - reference.float()).abs()
    max_abs_error = float(diff.max().item())
    max_rel_error = float((diff / reference.float().abs().clamp_min(1e-6)).max().item())
    skip_timing = args.backend == "torch_paged" and getattr(args, "skip_reference_timing", False)
    if skip_timing:
        log(f"[case] timing skipped for reference backend {case}")
        return {
            "backend": args.backend,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "block_size": block_size,
            "dtype": args.dtype,
            "num_q_heads": shapes.num_attention_heads,
            "num_kv_heads": shapes.num_key_value_heads,
            "head_dim": shapes.head_dim,
            "p50_latency_ms": None,
            "p95_latency_ms": None,
            "tokens_per_s": None,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
            "note": "reference timing skipped",
        }

    log(f"[case] warmup {case} iterations={args.warmup}")
    for _ in range(args.warmup):
        _ = paged_attention_decode(args.backend, q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
    if device.type == "cuda":
        torch.cuda.synchronize()

    log(f"[case] timing {case} iterations={args.repeat}")
    latencies = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        _ = paged_attention_decode(args.backend, q, k_cache, v_cache, block_tables, context_lens, scale, block_size)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((perf_counter() - start) * 1000.0)

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    log(f"[case] done {case} p50_ms={p50:.4f} p95_ms={p95:.4f} max_abs_error={max_abs_error:.4e}")
    return {
        "backend": args.backend,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "block_size": block_size,
        "dtype": args.dtype,
        "num_q_heads": shapes.num_attention_heads,
        "num_kv_heads": shapes.num_key_value_heads,
        "head_dim": shapes.head_dim,
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "tokens_per_s": batch_size / (p50 / 1000.0) if p50 else 0.0,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
    }


def write_outputs(rows: list[dict], args) -> None:
    if args.save_json:
        path = Path(args.save_json)
        log(f"[save] writing JSON results to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.save_md:
        path = Path(args.save_md)
        log(f"[save] writing Markdown results to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        body = [
            "# Attention Decode Benchmark",
            "",
            f"- model: `{args.model}`",
            f"- backend: `{args.backend}`",
            f"- dtype: `{args.dtype}`",
            f"- warmup: `{args.warmup}`",
            f"- repeat: `{args.repeat}`",
            "",
            markdown_table(rows),
            "",
            "This benchmark isolates single-token decode attention over a paged KV cache. "
            "It is a kernel/backend benchmark, not an end-to-end serving benchmark.",
            "",
        ]
        path.write_text("\n".join(body), encoding="utf-8")
    if not args.save_json and not args.save_md:
        log("[save] no output path requested; results printed to stdout only")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark decode-only paged attention backends.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--backend", default="triton_paged_decode")
    parser.add_argument("--seq-lens", default="1024,4096")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--block-sizes", default="16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--save-md")
    parser.add_argument("--save-json")
    parser.add_argument("--verbose", action="store_true", help="Print tensor shapes and model attention dimensions.")
    parser.add_argument(
        "--skip-reference-timing",
        action="store_true",
        help="Run correctness for torch_paged but skip its warmup/timing loop.",
    )
    args = parser.parse_args()

    log(f"[setup] loading model shapes from {args.model}")
    shapes = load_qwen_attention_shapes(args.model)
    verbose_case_details(args, shapes)
    rows = []
    for backend in parse_str_list(args.backend):
        log(f"[backend] start backend={backend}")
        case_args = argparse.Namespace(**vars(args))
        case_args.backend = backend
        for block_size in parse_int_list(args.block_sizes):
            for batch_size in parse_int_list(args.batch_sizes):
                for seq_len in parse_int_list(args.seq_lens):
                    try:
                        rows.append(run_case(case_args, shapes, batch_size, seq_len, block_size))
                    except Exception as exc:
                        log(
                            "[error] benchmark failed for "
                            f"{describe_case(case_args, shapes, batch_size, seq_len, block_size)}: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        raise
        log(f"[backend] done backend={backend}")
    log("[summary] benchmark table")
    print(markdown_table(rows), flush=True)
    write_outputs(rows, args)


if __name__ == "__main__":
    main()
