# Qwen3-4B RTX 5090 Kernel Policy Report

## Inputs

- Kernel results: `results/rtx5090_qwen3_4b/kernels_qwen3_4b_5090.md`
- CUDA GEMM results: `results/rtx5090_qwen3_4b/cuda_gemm_qwen3_4b_5090.md`
- E2E results: `results/rtx5090_qwen3_4b/e2e_qwen3_4b_5090.md`

## E2E Summary

This E2E summary is a policy-report snapshot. For the latest standalone fork-only repeat3 E2E result,
see `e2e_qwen3_4b_5090_repeat3.md`. For upstream-vs-fork speedup claims, see
`upstream_vs_fork_qwen3_4b_5090.md`.

| Metric | Value |
| --- | --- |
| model | ../models/Qwen3-4B |
| gpu | NVIDIA GeForce RTX 5090 |
| prompt_len | 512 |
| num_prompts | 4 |
| max_tokens | 128 |
| elapsed_s | 2.5879 |
| ttft_s | 0.2811 |
| prefill_time_s | 0.0856 |
| decode_time_s | 2.4992 |
| decode_tokens_per_s | 203.2630 |
| itl_ms_avg | 4.9197 |
| decode_step_ms_p50 | 18.1949 |
| decode_step_ms_p95 | 18.9070 |
| peak_gpu_memory_gb | 27.5493 |
| num_kvcache_blocks | 563 |
| max_used_blocks | 12 |
| max_block_utilization | 0.0213 |
| kv_cache_dtype | bfloat16 |
| linear_backend | torch |

## Kernel Wins

| Kernel | Shape | Baseline | Triton | Speedup | Correctness |
| --- | --- | --- | --- | --- | --- |
| RMSNorm | N=1,D=2560 | 84.55 us | 34.79 us | 2.43x | pass max=0, mean=0 |
| AddRMSNorm | N=1,D=2560 | 97.85 us | 52.23 us | 1.87x | pass max=0, mean=0 |
| RMSNorm | N=16,D=2560 | 94.17 us | 34.79 us | 2.71x | pass max=0, mean=0 |
| AddRMSNorm | N=16,D=2560 | 134.64 us | 53.30 us | 2.53x | pass max=0, mean=0 |
| RMSNorm | N=128,D=2560 | 100.21 us | 34.78 us | 2.88x | pass max=0, mean=0 |
| AddRMSNorm | N=128,D=2560 | 107.95 us | 52.91 us | 2.04x | pass max=0, mean=0 |
| RMSNorm | N=1024,D=2560 | 96.16 us | 35.43 us | 2.71x | pass max=0.00781, mean=5.81e-08 |
| AddRMSNorm | N=1024,D=2560 | 110.77 us | 53.70 us | 2.06x | pass max=0, mean=0 |
| RMSNorm | N=4096,D=2560 | 96.14 us | 35.23 us | 2.73x | pass max=0, mean=0 |
| AddRMSNorm | N=4096,D=2560 | 110.66 us | 56.36 us | 1.96x | pass max=0, mean=0 |
| SiluAndMul | N=1,D=9728 | 52.62 us | 35.77 us | 1.47x | pass max=0, mean=0 |
| SiluAndMul | N=16,D=9728 | 57.89 us | 36.14 us | 1.60x | pass max=0, mean=0 |
| SiluAndMul | N=128,D=9728 | 58.45 us | 35.81 us | 1.63x | pass max=0, mean=0 |
| RoPE | N=16,QH=32,KVH=8,D=128 | 106.71 us | 75.44 us | 1.41x | pass max=0, mean=0 |
| RoPE | N=128,QH=32,KVH=8,D=128 | 82.60 us | 74.84 us | 1.10x | pass max=0, mean=0 |
| RoPE | N=1024,QH=32,KVH=8,D=128 | 83.02 us | 76.78 us | 1.08x | pass max=0, mean=0 |

## Kernel Regressions

| Kernel | Shape | Baseline | Triton | Speedup | Correctness |
| --- | --- | --- | --- | --- | --- |
| RoPE | N=4096,QH=32,KVH=8,D=128 | 84.35 us | 125.60 us | 0.67x | pass max=0, mean=0 |
| KVCacheStore | N=1,KVH=8,D=128 | 24.12 us | 28.08 us | 0.86x | pass max=0, mean=0 |
| KVCacheStore | N=16,KVH=8,D=128 | 24.45 us | 28.04 us | 0.87x | pass max=0, mean=0 |
| KVCacheStore | N=128,KVH=8,D=128 | 24.15 us | 28.40 us | 0.85x | pass max=0, mean=0 |
| KVCacheStore | N=512,KVH=8,D=128 | 23.99 us | 28.15 us | 0.85x | pass max=0, mean=0 |
| KVCacheStore | N=2048,KVH=8,D=128 | 24.15 us | 28.66 us | 0.84x | pass max=0, mean=0 |
| Linear-QKV | M=1,K=2560,N=6144 | 13.49 us | 40.99 us | 0.33x | pass max=7.63e-06, mean=1.24e-09 |
| Linear-QKV | M=4,K=2560,N=6144 | 12.37 us | 43.78 us | 0.28x | pass max=0.5, mean=0.00016 |
| Linear-QKV | M=8,K=2560,N=6144 | 13.39 us | 50.85 us | 0.26x | pass max=0.5, mean=0.000171 |
| Linear-QKV | M=16,K=2560,N=6144 | 15.63 us | 112.82 us | 0.14x | pass max=0.5, mean=0.000105 |
| Linear-O_proj | M=1,K=4096,N=2560 | 9.90 us | 64.38 us | 0.15x | pass max=0, mean=0 |
| Linear-O_proj | M=4,K=4096,N=2560 | 15.71 us | 64.61 us | 0.24x | pass max=0.5, mean=0.000137 |
| Linear-O_proj | M=8,K=4096,N=2560 | 16.00 us | 65.52 us | 0.24x | pass max=1, mean=0.000271 |
| Linear-O_proj | M=16,K=4096,N=2560 | 17.11 us | 71.97 us | 0.24x | pass max=1, mean=0.000224 |
| Linear-Gate/Up | M=1,K=2560,N=19456 | 25.03 us | 51.50 us | 0.49x | pass max=0.000488, mean=2.55e-08 |
| Linear-Gate/Up | M=4,K=2560,N=19456 | 29.46 us | 71.07 us | 0.41x | pass max=1, mean=0.000105 |
| Linear-Gate/Up | M=8,K=2560,N=19456 | 33.05 us | 136.66 us | 0.24x | pass max=0.5, mean=0.000108 |
| Linear-Gate/Up | M=16,K=2560,N=19456 | 37.73 us | 252.72 us | 0.15x | pass max=1, mean=9.89e-05 |
| Linear-Down | M=1,K=9728,N=2560 | 14.32 us | 153.04 us | 0.09x | pass max=0.25, mean=0.000105 |
| Linear-Down | M=4,K=9728,N=2560 | 16.21 us | 153.54 us | 0.11x | pass max=2, mean=0.127 |
| Linear-Down | M=8,K=9728,N=2560 | 35.60 us | 155.53 us | 0.23x | pass max=1, mean=0.00103 |
| Linear-Down | M=16,K=9728,N=2560 | 38.44 us | 170.22 us | 0.23x | pass max=2, mean=0.000708 |

## CUDA GEMM Worklog Summary

Custom CUDA GEMM is treated as a research baseline. Rows below are cases where naive/tiled CUDA is slower than cuBLAS.

| Shape | M | K | N | torch.matmul | CUDA naive | CUDA tiled | Naive vs torch | Tiled vs torch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| qkv_decode | 1 | 2560 | 6144 | 15.92 us / 1975.4 GFLOP/s | 58.01 us / 542.3 GFLOP/s | 75.85 us / 414.7 GFLOP/s | 0.27x | 0.21x |
| o_proj_decode | 1 | 4096 | 2560 | 13.47 us / 1557.1 GFLOP/s | 92.17 us / 227.5 GFLOP/s | 56.00 us / 374.5 GFLOP/s | 0.15x | 0.24x |
| gate_up_decode | 1 | 2560 | 19456 | 123.28 us / 808.1 GFLOP/s | 250.04 us / 398.4 GFLOP/s | 198.43 us / 502.0 GFLOP/s | 0.49x | 0.62x |
| down_proj_decode | 1 | 9728 | 2560 | 34.00 us / 1465.0 GFLOP/s | 395.85 us / 125.8 GFLOP/s | 182.27 us / 273.3 GFLOP/s | 0.09x | 0.19x |
| qkv_small_batch | 16 | 2560 | 6144 | 38.36 us / 13122.1 GFLOP/s | 116.66 us / 4314.4 GFLOP/s | 76.79 us / 6554.5 GFLOP/s | 0.33x | 0.50x |
| o_proj_small_batch | 16 | 4096 | 2560 | 28.32 us / 11849.3 GFLOP/s | 115.85 us / 2896.5 GFLOP/s | 56.53 us / 5935.3 GFLOP/s | 0.24x | 0.50x |
| gate_up_small_batch | 16 | 2560 | 19456 | 126.04 us / 12645.5 GFLOP/s | 321.77 us / 4953.3 GFLOP/s | 202.12 us / 7885.8 GFLOP/s | 0.39x | 0.62x |
| down_proj_small_batch | 16 | 9728 | 2560 | 51.18 us / 15571.3 GFLOP/s | 476.28 us / 1673.2 GFLOP/s | 195.66 us / 4073.1 GFLOP/s | 0.11x | 0.26x |
| qkv_prefill | 256 | 2560 | 6144 | 141.03 us / 57100.7 GFLOP/s | 1126.43 us / 7149.2 GFLOP/s | 817.26 us / 9853.8 GFLOP/s | 0.13x | 0.17x |
| gate_up_prefill | 256 | 2560 | 19456 | 438.92 us / 58099.8 GFLOP/s | 4209.07 us / 6058.7 GFLOP/s | 2688.49 us / 9485.4 GFLOP/s | 0.10x | 0.16x |
| down_proj_prefill | 256 | 9728 | 2560 | 258.26 us / 49372.3 GFLOP/s | 2008.55 us / 6348.2 GFLOP/s | 1378.65 us / 9248.7 GFLOP/s | 0.13x | 0.19x |

## Recommended Default Policy

| Kernel | Policy |
| --- | --- |
| RMSNorm | "triton" |
| AddRMSNorm | "triton" |
| SiluAndMul | {"N<=128": "triton", "N>=1024": "torch"} |
| RoPE | {"N<=1024": "triton", "N>=4096": "torch"} |
| KVCacheStore | "1d_triton" |
| Linear | "torch_cublas" |
| CUDA_GEMM | "research_only" |

## Negative Results Explanation

- RMSNorm/AddRMSNorm use `triton` / `triton` because the measured Triton path is consistently faster.
- SiLU-and-Mul policy is `{'N<=128': 'triton', 'N>=1024': 'torch'}` because large prefill shapes can erase the small-batch Triton win.
- RoPE policy is `{'N<=1024': 'triton', 'N>=4096': 'torch'}` because the Triton path is shape-sensitive.
- KVCacheStore remains `1d_triton` because the experimental 2D store regressed versus the current 1D Triton store.
- Linear remains `torch_cublas` because cuBLAS is faster than the educational Triton GEMV on Qwen3-4B shapes.
- CUDA_GEMM is `research_only` because naive/tiled CUDA kernels are worklog baselines, not production replacements.

## Runtime Scope

This policy is a report artifact only. It is not wired into nano-vLLM runtime dispatch, and the default serving path remains FlashAttention 2, model-dtype KV cache, and cuBLAS Linear.
