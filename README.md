<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Triton kernels, CUDA graph, etc.
* 🧪 **Qwen3-4B / RTX 5090 Profiling** - Model-shape-aware kernel, CUDA GEMM, and end-to-end benchmarks

## Installation

For local development, clone this fork and install it in editable mode:

```bash
git clone https://github.com/Creativecole/nano-vllm.git
cd nano-vllm
pip install -e .
```

Or install directly from this fork:

```bash
pip install git+https://github.com/Creativecole/nano-vllm.git
```

This fork keeps the serving path on FlashAttention 2, model-dtype KV cache, and cuBLAS Linear for
reproducible results. Experimental kernels are developed behind benchmarks before they are considered
for the default inference path.

The fork is intentionally a lightweight extension of nano-vLLM rather than a rewrite. Kernel backend
configuration is exposed through `Config`, while the default runtime remains conservative:

| Component | Default | Notes |
|---|---|---|
| Attention | FlashAttention 2 | Existing paged KV-cache decode path |
| KV cache | model dtype | Usually BF16 for Qwen3 checkpoints |
| Linear | cuBLAS / `F.linear` | Custom GEMM kernels stay in benchmarks until proven faster |
| RMSNorm / SiLU / RoPE | Triton | Covered by correctness tests and microbenchmarks |

## Model Download

To download the Qwen3-0.6B example model manually, use:

```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

Larger compatible dense Qwen3 CausalLM checkpoints can also be used by passing their local directory to
`LLM`, for example `/path/to/Qwen3-4B`. Hybrid checkpoints with `linear_attn` weights need a separate
model adapter and are intentionally rejected by this fork.

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM(
    "/YOUR/MODEL/PATH",
    enforce_eager=True,
    tensor_parallel_size=1,
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |

## Qwen3-4B / RTX 5090 Kernel Work

This fork focuses on measurable Triton kernel work first: every replacement path is checked against a
PyTorch or torch.compile baseline before being used as a default. The current RTX 5090 / Qwen3-0.6B
microbenchmarks show clear wins for normalization and activation kernels, while KV-cache store variants
and GEMV experiments are kept out of the default path when they do not beat the baseline.

Measured on RTX 5090 with `python bench_kernels.py --min-run-time 1.0 --skip-sampler`:

| Kernel | Baseline | Triton result | Default decision |
|---|---|---:|---|
| RMSNorm | torch.compile | up to 2.95x faster | Enabled |
| Add + RMSNorm | torch.compile | up to 2.21x faster | Enabled |
| SiLU-and-Mul | torch.compile | up to 1.83x faster | Enabled |
| RoPE | torch.compile | up to 1.18x faster | Enabled |
| KV-cache store 2D grid | 1D Triton store | ~0.84-0.86x | Experimental only |
| Linear GEMV | `torch.nn.functional.linear` / cuBLAS | ~0.27-0.36x | Disabled by default |

The repo also includes a standalone CUDA C++ GEMM worklog benchmark. It starts with a naive FP32 GEMM
kernel and a shared-memory tiled GEMM kernel, then compares both against `torch.matmul` / cuBLAS on
Qwen3-like linear-layer shapes. This is kept separate from `LinearBase` because cuBLAS remains the
production baseline until a custom BF16 Tensor Core kernel proves faster.

Useful commands:

```bash
pytest tests/test_kernels.py
python bench_kernels.py --min-run-time 1.0 --skip-sampler --output kernels_5090_qwen3_0.6b.md
python bench_kernels.py --model /path/to/Qwen3-4B --min-run-time 1.0 --skip-sampler --output kernels_qwen3_4b_5090.md
python bench_cuda_gemm.py --min-run-time 1.0 --output cuda_gemm_5090.md
python bench_cuda_gemm.py --model /path/to/Qwen3-4B --min-run-time 1.0 --output cuda_gemm_qwen3_4b_5090.md
python bench_e2e.py --model /path/to/Qwen3-4B --prompt-len 512 --num-prompts 4 --max-tokens 128 --enforce-eager --output e2e_qwen3_4b_5090.md
python analyze_kernel_results.py --kernel-results kernels_qwen3_4b_5090.md --cuda-gemm-results cuda_gemm_qwen3_4b_5090.md --e2e-results e2e_qwen3_4b_5090.md --output KERNEL_POLICY_REPORT.md --policy-output kernel_policy_5090.json
```

`bench_kernels.py` prints a Markdown summary table with median latency, speedup, and correctness status.
When `--model` is provided, it derives Qwen layer shapes from the checkpoint config: RMSNorm, AddRMSNorm,
SiLU-and-Mul, RoPE, KV-cache store, QKV/O projection, and MLP gate/up/down linear shapes.

`bench_cuda_gemm.py` JIT-compiles the CUDA extension in `csrc/cuda_gemm_kernel.cu` and reports latency,
GFLOP/s, speedup versus cuBLAS, and correctness for each Qwen linear-layer shape. It is a CUDA GEMM
worklog, not a production `LinearBase` replacement. The next optimization steps are vectorized global
loads, register tiling, BF16 Tensor Core MMA, and comparison with Triton `tl.dot` and cuBLAS.

`bench_e2e.py` runs real nano-vLLM generation through the scheduler loop and reports elapsed time, TTFT,
prefill/decode time, decode tokens/s, ITL, peak GPU memory, KV-cache block counts, block utilization,
prefix-cache hit/miss counters, and the active backend configuration.

`analyze_kernel_results.py` turns the three benchmark result files into a Markdown report and
`kernel_policy_5090.json`. The generated policy is an analysis artifact only; it is not wired into
runtime dispatch.

Future work includes model-shape-aware kernel autotuning, BF16 Tensor Core GEMM experiments, and
additional scheduler / KV-cache observability for Qwen3-4B workloads.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
