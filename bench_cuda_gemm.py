"""
CUDA C++ GEMM microbenchmark for the nano-vLLM fork.

This is intentionally separate from the default LinearBase path. It compares
educational CUDA kernels against torch.matmul/cuBLAS on shapes that resemble
Qwen3 linear layers, and records both positive and negative results.
"""

import argparse
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark
from torch.utils.cpp_extension import load

from nanovllm.utils.model_shapes import ModelShapes


SUMMARY = []


def load_extension():
    root = Path(__file__).resolve().parent
    return load(
        name="nanovllm_cuda_gemm",
        sources=[str(root / "csrc" / "cuda_gemm_kernel.cu")],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )


def time_fn(fn, min_run_time):
    return benchmark.Timer(stmt="fn()", globals={"fn": fn}).blocked_autorange(min_run_time=min_run_time)


def gflops(m, n, k, seconds):
    return (2.0 * m * n * k) / seconds / 1e9


def assert_close(name, actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-2)
    diff = (actual - expected).abs()
    return f"pass max={diff.max().item():.3g}, mean={diff.mean().item():.3g}"


def add_summary(shape_name, m, k, n, baseline, naive, tiled, correctness):
    SUMMARY.append(
        {
            "shape": shape_name,
            "m": m,
            "k": k,
            "n": n,
            "torch_us": baseline.median * 1e6,
            "naive_us": naive.median * 1e6,
            "tiled_us": tiled.median * 1e6,
            "torch_gflops": gflops(m, n, k, baseline.median),
            "naive_gflops": gflops(m, n, k, naive.median),
            "tiled_gflops": gflops(m, n, k, tiled.median),
            "naive_speedup": baseline.median / naive.median,
            "tiled_speedup": baseline.median / tiled.median,
            "correctness": correctness,
        }
    )


def print_summary():
    lines = [
        "| Shape | M | K | N | torch.matmul | CUDA naive | CUDA tiled | Naive vs torch | Tiled vs torch | Correctness |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in SUMMARY:
        lines.append(
            f"| {row['shape']} | {row['m']} | {row['k']} | {row['n']} | "
            f"{row['torch_us']:.2f} us / {row['torch_gflops']:.1f} GFLOP/s | "
            f"{row['naive_us']:.2f} us / {row['naive_gflops']:.1f} GFLOP/s | "
            f"{row['tiled_us']:.2f} us / {row['tiled_gflops']:.1f} GFLOP/s | "
            f"{row['naive_speedup']:.2f}x | {row['tiled_speedup']:.2f}x | {row['correctness']} |"
        )
    text = "\n".join(lines)
    print("\nMarkdown summary table")
    print("=" * 70)
    print(text)
    return text


def load_shapes(model_path: str | None) -> ModelShapes:
    if model_path:
        return ModelShapes.from_model(model_path)
    return ModelShapes.default()


def main():
    parser = argparse.ArgumentParser(description="Benchmark CUDA C++ GEMM kernels against torch.matmul/cuBLAS.")
    parser.add_argument("--model", type=str, default=None, help="Optional model path used to derive Qwen linear shapes.")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_cuda_gemm.py")

    ext = load_extension()
    torch.set_default_device("cuda")
    torch.manual_seed(0)

    print("=" * 70)
    print("nano-vLLM CUDA GEMM Microbenchmarks")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    shapes = load_shapes(args.model)
    print(f"Shapes: {shapes.model_name} ({shapes.describe()})")

    configs = [
        ("qkv_decode", 1, shapes.hidden_size, shapes.qkv_dim),
        ("o_proj_decode", 1, shapes.o_proj_in, shapes.hidden_size),
        ("gate_up_decode", 1, shapes.hidden_size, shapes.intermediate_size * 2),
        ("down_proj_decode", 1, shapes.intermediate_size, shapes.hidden_size),
        ("qkv_small_batch", 16, shapes.hidden_size, shapes.qkv_dim),
        ("o_proj_small_batch", 16, shapes.o_proj_in, shapes.hidden_size),
        ("gate_up_small_batch", 16, shapes.hidden_size, shapes.intermediate_size * 2),
        ("down_proj_small_batch", 16, shapes.intermediate_size, shapes.hidden_size),
        ("qkv_prefill", 256, shapes.hidden_size, shapes.qkv_dim),
        ("gate_up_prefill", 256, shapes.hidden_size, shapes.intermediate_size * 2),
        ("down_proj_prefill", 256, shapes.intermediate_size, shapes.hidden_size),
    ]

    results = []
    for name, m, k, n in configs:
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)

        expected = torch.matmul(a, b)
        naive_out = ext.matmul_naive(a, b)
        tiled_out = ext.matmul_tiled(a, b)
        torch.cuda.synchronize()
        assert_close(f"{name} naive", naive_out, expected)
        correctness = assert_close(f"{name} tiled", tiled_out, expected)

        for _ in range(5):
            torch.matmul(a, b)
            ext.matmul_naive(a, b)
            ext.matmul_tiled(a, b)
        torch.cuda.synchronize()

        torch_result = time_fn(lambda: torch.matmul(a, b), args.min_run_time)
        naive_result = time_fn(lambda: ext.matmul_naive(a, b), args.min_run_time)
        tiled_result = time_fn(lambda: ext.matmul_tiled(a, b), args.min_run_time)
        results.extend([torch_result, naive_result, tiled_result])
        add_summary(name, m, k, n, torch_result, naive_result, tiled_result, correctness)

        print(
            f"{name:>16} M={m:<4} K={k:<5} N={n:<5} "
            f"torch={torch_result.median * 1e6:8.2f}us "
            f"naive={naive_result.median * 1e6:8.2f}us "
            f"tiled={tiled_result.median * 1e6:8.2f}us"
        )

    summary = print_summary()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary + "\n", encoding="utf-8")
        print(f"\nSaved Markdown summary to {output_path}")


if __name__ == "__main__":
    main()
