import argparse

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import triton
import triton.language as tl


@triton.jit
def _fp8_gemm_kernel(
    x_ptr,
    w_ptr,
    x_scale_ptr,
    w_scale_ptr,
    out_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_x_m: tl.constexpr,
    stride_w_n: tl.constexpr,
    stride_out_m: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_scale = tl.load(x_scale_ptr)
    w_scale = tl.load(w_scale_ptr)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_x_m + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * stride_w_n + k[:, None],
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w)

    acc = acc * x_scale * w_scale
    tl.store(
        out_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def quantize_fp8_per_tensor(x: torch.Tensor):
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    scale = x.abs().max().float().clamp_min(1e-6) / fp8_max
    q = torch.clamp(x.float() / scale, -fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale


def fp8_gemm(x_fp8, w_fp8, x_scale, w_scale, out_dtype=torch.bfloat16):
    M, K = x_fp8.shape
    N = w_fp8.shape[0]
    out = torch.empty((M, N), device=x_fp8.device, dtype=out_dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
    _fp8_gemm_kernel[grid](
        x_fp8,
        w_fp8,
        x_scale,
        w_scale,
        out,
        M,
        N,
        K,
        x_fp8.stride(0),
        w_fp8.stride(0),
        out.stride(0),
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="Standalone Triton FP8 GEMM microbenchmark.")
    parser.add_argument("--min-run-time", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")
    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("torch.float8_e4m3fn is required.")

    configs = [
        ("decode_qkv", 1, 1024, 1280),
        ("decode_mlp", 1, 1024, 5632),
        ("batched_qkv", 128, 1024, 1280),
        ("prefill_mlp", 1024, 1024, 5632),
    ]
    results = []
    for name, M, K, N in configs:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        x_fp8, x_scale = quantize_fp8_per_tensor(x)
        w_fp8, w_scale = quantize_fp8_per_tensor(w)
        for _ in range(5):
            F.linear(x, w)
            fp8_gemm(x_fp8, w_fp8, x_scale, w_scale)
        results.append(
            benchmark.Timer(
                stmt="F.linear(x, w)",
                globals={"F": F, "x": x, "w": w},
                label=name,
                sub_label=f"M={M},K={K},N={N}",
                description="BF16 F.linear",
            ).blocked_autorange(min_run_time=args.min_run_time)
        )
        results.append(
            benchmark.Timer(
                stmt="fp8_gemm(x_fp8, w_fp8, x_scale, w_scale)",
                globals={
                    "fp8_gemm": fp8_gemm,
                    "x_fp8": x_fp8,
                    "w_fp8": w_fp8,
                    "x_scale": x_scale,
                    "w_scale": w_scale,
                },
                label=name,
                sub_label=f"M={M},K={K},N={N}",
                description="Triton FP8 tl.dot",
            ).blocked_autorange(min_run_time=args.min_run_time)
        )
    benchmark.Compare(results).print()


if __name__ == "__main__":
    main()
