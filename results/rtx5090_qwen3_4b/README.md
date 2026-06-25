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
| `profile_qwen3_4b_5090.md` | PyTorch profiler top-ops summary |
| `profile_qwen3_4b_5090.json` | Chrome trace exported by PyTorch profiler; generated locally, not tracked |
| `upstream_vs_fork_qwen3_4b_5090.md` | Upstream nano-vLLM vs fork e2e comparison |
| `prefix_cache_qwen3_4b_5090.md` | Prefix-cache workload benchmark |

## Upstream vs Fork E2E

Measured with Qwen3-4B, prompt length 512, 4 prompts, 128 generated tokens, `--enforce-eager`,
1 warmup run, and 3 measured runs.

| Metric | Upstream nano-vLLM | This fork | Delta |
|---|---:|---:|---:|
| Elapsed time | 5.5611 s | 3.2961 s | 1.687x lower |
| Decode tokens/s | 92.9235 | 160.2852 | 1.725x higher |
| Average ITL | 10.7815 ms | 6.3258 ms | 1.704x lower |
| Decode step p95 | 45.6434 ms | 25.9239 ms | 1.761x lower |
| Peak GPU memory | 27.4288 GB | 27.3754 GB | 1.002x lower |

## Prefix Cache Workloads

| Workload | TTFT | Decode tokens/s | Prefix hit rate | Interpretation |
|---|---:|---:|---:|---|
| no shared prefix | 374.4 ms | 115.9 | 0.00 | Distinct prompts do not reuse full KV blocks |
| shared system prompt | 210.7 ms | 122.6 | 1.00 | Shared system blocks reduce prefill cost |
| shared few-shot prefix | 65.3 ms | 168.8 | 1.00 | Longer shared prefix gives the largest TTFT win |
| long prompt, short decode | 190.6 ms | 166.7 | 0.80 | Long prompts benefit from prefix reuse |
| short prompt, long decode | 54.6 ms | 167.0 | 0.00 | Short prompts have too few full blocks to reuse |

Compared with no shared prefix, the shared few-shot workload reduces TTFT by about 5.7x and increases
decode throughput by about 1.46x. This is why the fork exposes prefix-cache hit/miss counters and KV
block utilization instead of treating KV cache as an invisible implementation detail.

## Profiler Evidence

The PyTorch profiler run captures 64 scheduler steps: 1 prefill step and 63 decode steps. The trace
shows that Qwen3-4B decode kernel self-time is dominated by BF16 Linear/GEMM work:

| Profiler item | CUDA time | Calls | Interpretation |
|---|---:|---:|---|
| Linear/GEMM CUDA kernels | 421.8 ms | 11,548 | Main decode kernel bottleneck |
| FlashAttention kernels | 30.2 ms | 4,572 | Attention is much smaller than GEMM |
| Triton SiLU/RMSNorm/RoPE/store kernels | 35.5 ms | 18,496 | Useful but smaller contributors |
| KV-cache store kernel | 2.7 ms | 2,304 | KV store is not the current bottleneck |

This motivates keeping cuBLAS Linear as the production default while using the CUDA GEMM worklog as
the next research track: vectorized loads, register tiling, and BF16 Tensor Core MMA.

## Headline Findings

| Area | Result |
|---|---|
| RMSNorm/AddRMSNorm | Triton is consistently faster on Qwen3-4B shapes |
| SiLU-and-Mul | Triton wins on small/medium token counts, but large prefill shapes are near parity |
| RoPE | Triton is shape-sensitive; it wins for medium shapes and regresses at very large N |
| KV-cache store | Current 1D Triton store remains better than the experimental 2D store |
| Linear/GEMV | cuBLAS remains the correct default on Qwen3-4B linear shapes |
| CUDA GEMM | Naive/tiled CUDA kernels are research baselines, not production replacements |
| E2E fork delta | 1.73x higher decode tokens/s than upstream on the measured workload |
| Prefix cache | Shared few-shot prefix reduces TTFT by about 5.7x versus no shared prefix |

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

python profile_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 128 \
  --enforce-eager \
  --profile-steps 64 \
  --profile-memory \
  --record-shapes \
  --trace-output results/rtx5090_qwen3_4b/profile_qwen3_4b_5090.json \
  --summary-output results/rtx5090_qwen3_4b/profile_qwen3_4b_5090.md

python compare_upstream.py \
  --upstream-repo ../nano-vllm-upstream \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 128 \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --output results/rtx5090_qwen3_4b/upstream_vs_fork_qwen3_4b_5090.md

python bench_prefix_cache.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 128 \
  --enforce-eager \
  --save-md results/rtx5090_qwen3_4b/prefix_cache_qwen3_4b_5090.md \
  --save-json results/rtx5090_qwen3_4b/prefix_cache_qwen3_4b_5090.json

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
| Prefix-cache benchmark | Not included | Dedicated prompt-sharing workloads with hit/miss counters |

## Runtime Scope

The generated policy is a reporting artifact. It is not wired into runtime dispatch.
The default serving path stays conservative: FlashAttention 2, model-dtype KV cache, and cuBLAS Linear.

## Next Steps

- Implement optional backend dispatch only for safe layer kernels such as RMSNorm, activation, and RoPE.
- Add a BF16 Tensor Core GEMM experiment before considering any Linear runtime integration.
- Add Nsight Systems screenshots for launch overhead and decode-step timeline visualization.
