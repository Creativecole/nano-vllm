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

from nanovllm.backends import prefill_attention
from nanovllm.kernels.attention.torch_attention import torch_sdpa_prefill
from nanovllm.utils.shapes import load_qwen_attention_shapes


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}[name.lower()]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def markdown_table(rows: list[dict]) -> str:
    if not rows:
        return "_No rows generated._"
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def run_backend(backend, q, k, v, cu_seqlens, scale):
    return prefill_attention(backend, q, k, v, cu_seqlens, scale)


def run_case(args, shapes, backend: str, batch_size: int, seq_len: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend == "flash_attn" and device.type != "cuda":
        raise RuntimeError("flash_attn benchmark requires CUDA")
    dtype = dtype_from_name(args.dtype)
    total_tokens = batch_size * seq_len
    q = torch.randn(total_tokens, shapes.num_attention_heads, shapes.head_dim, device=device, dtype=dtype)
    k = torch.randn(total_tokens, shapes.num_key_value_heads, shapes.head_dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    cu_seqlens = torch.arange(0, total_tokens + 1, seq_len, device=device, dtype=torch.int32)
    scale = shapes.head_dim ** -0.5
    ref = torch_sdpa_prefill(q, k, v, cu_seqlens, scale)
    out = run_backend(backend, q, k, v, cu_seqlens, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    diff = (out.float() - ref.float()).abs()
    for _ in range(args.warmup):
        _ = run_backend(backend, q, k, v, cu_seqlens, scale)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latencies = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = perf_counter()
        _ = run_backend(backend, q, k, v, cu_seqlens, scale)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((perf_counter() - start) * 1000.0)
    p50 = percentile(latencies, 50)
    return {
        "backend": backend,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "total_tokens": total_tokens,
        "dtype": args.dtype,
        "p50_latency_ms": p50,
        "p95_latency_ms": percentile(latencies, 95),
        "tokens_per_s": total_tokens / (p50 / 1000.0) if p50 else 0.0,
        "max_abs_error": float(diff.max().item()),
        "max_rel_error": float((diff / ref.float().abs().clamp_min(1e-6)).max().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark prefill attention backends.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--backends", default="torch_sdpa,flash_attn")
    parser.add_argument("--seq-lens", default="512,1024")
    parser.add_argument("--batch-sizes", default="1,4")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--save-md")
    parser.add_argument("--save-json")
    args = parser.parse_args()
    shapes = load_qwen_attention_shapes(args.model)
    rows = []
    for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
        for batch_size in parse_int_list(args.batch_sizes):
            for seq_len in parse_int_list(args.seq_lens):
                rows.append(run_case(args, shapes, backend, batch_size, seq_len))
    print(markdown_table(rows))
    if args.save_json:
        path = Path(args.save_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if args.save_md:
        path = Path(args.save_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Attention Prefill Benchmark\n\n" + markdown_table(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
