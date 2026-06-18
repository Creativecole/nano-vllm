# Qwen3-4B RTX 5090 Results

This directory contains curated benchmark artifacts for the nano-vLLM fork on a single RTX 5090.
The goal is to document model-shape-aware kernel benchmarking, CUDA GEMM worklog results, end-to-end
serving metrics, and the derived kernel policy without changing the conservative runtime path.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Model | Qwen3-4B |
| Runtime default | FlashAttention 2 + model-dtype KV cache + cuBLAS Linear |
| Custom kernels | Triton RMSNorm/AddRMSNorm/SiLU/RoPE, Triton KV-store variants, standalone CUDA GEMM worklog |

## Artifacts

| File | Purpose |
|---|---|
| `kernels_qwen3_4b_5090.md` | Model-shape-aware Triton/cuBLAS microbenchmark results |
| `cuda_gemm_qwen3_4b_5090.md` | CUDA naive/tiled GEMM worklog versus cuBLAS |
| `e2e_qwen3_4b_5090.md` | Single-run end-to-end TTFT/decode/KV-cache metrics |
| `e2e_qwen3_4b_5090_repeat3.md` | Repeat-run end-to-end results with mean/p50/p95 |
| `KERNEL_POLICY_REPORT.md` | Generated benchmark analysis report |
| `kernel_policy_5090.json` | Generated kernel policy artifact |

## Headline Findings

| Area | Result |
|---|---|
| RMSNorm/AddRMSNorm | Triton is consistently faster on Qwen3-4B shapes |
| SiLU-and-Mul | Triton wins on small/medium token counts, but large prefill shapes are near parity |
| RoPE | Triton is shape-sensitive; it wins for medium shapes and regresses at very large N |
| KV-cache store | Current 1D Triton store remains better than the experimental 2D store |
| Linear/GEMV | cuBLAS remains the correct default on Qwen3-4B linear shapes |
| CUDA GEMM | Naive/tiled CUDA kernels are research baselines, not production replacements |

## Reproduction

```bash
python bench_kernels.py \
  --model ../models/Qwen3-4B \
  --min-run-time 1.0 \
  --skip-sampler \
  --output results/rtx5090_qwen3_4b/kernels_qwen3_4b_5090.md

python bench_cuda_gemm.py \
  --model ../models/Qwen3-4B \
  --min-run-time 1.0 \
  --output results/rtx5090_qwen3_4b/cuda_gemm_qwen3_4b_5090.md

python bench_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 128 \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --output results/rtx5090_qwen3_4b/e2e_qwen3_4b_5090_repeat3.md

python analyze_kernel_results.py \
  --kernel-results results/rtx5090_qwen3_4b/kernels_qwen3_4b_5090.md \
  --cuda-gemm-results results/rtx5090_qwen3_4b/cuda_gemm_qwen3_4b_5090.md \
  --e2e-results results/rtx5090_qwen3_4b/e2e_qwen3_4b_5090.md \
  --output results/rtx5090_qwen3_4b/KERNEL_POLICY_REPORT.md \
  --policy-output results/rtx5090_qwen3_4b/kernel_policy_5090.json
```

## Upstream Delta

| Area | Upstream nano-vLLM | This fork |
|---|---|---|
| Kernel benchmark shapes | Mostly fixed script shapes | Model-shape-aware Qwen3-4B shapes |
| E2E profiling | Basic throughput benchmark | TTFT, prefill/decode time, ITL, KV-cache usage, peak memory |
| CUDA GEMM | Not included | Standalone CUDA GEMM worklog benchmark against cuBLAS |
| Observability | Minimal runtime counters | Block utilization, max block usage, prefix-cache hit/miss counters |
| Policy artifact | Not included | Generated Markdown report and JSON kernel policy |

## Runtime Scope

The generated policy is a reporting artifact. It is not wired into runtime dispatch.
The default serving path stays conservative: FlashAttention 2, model-dtype KV cache, and cuBLAS Linear.

## Next Steps

- Add Nsight Systems / PyTorch profiler evidence for the Qwen3-4B decode path.
- Implement optional backend dispatch only for safe layer kernels such as RMSNorm, activation, and RoPE.
- Add a BF16 Tensor Core GEMM experiment before considering any Linear runtime integration.
