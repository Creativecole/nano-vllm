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
* 🧪 **RTX 5090 Triton Benchmarks** - Benchmark-driven kernel work with correctness checks and measured speedups

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

This fork keeps the default serving path on FlashAttention 2 and BF16 KV cache for reproducible results.
Experimental FP8 KV-cache code is kept behind `kv_cache_dtype="fp8_e4m3"` and requires a FlashAttention
build whose `flash_attn_with_kvcache` exposes `k_descale` / `v_descale`.

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
    kv_cache_dtype="bf16",
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

## RTX 5090 Triton Kernel Work

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

Useful commands:

```bash
pytest tests/test_kernels.py
python bench_kernels.py --min-run-time 1.0 --skip-sampler --output kernels_5090_qwen3_0.6b.md
python bench_fp8_kvcache.py --model /path/to/Qwen3-0.6B --max-model-len 4096
```

`bench_kernels.py` prints a Markdown summary table with median latency, speedup, and correctness status.
The benchmark is meant to document both successful kernel substitutions and negative results, which keeps
the default inference path conservative.

## Experimental FP8 KV Cache

FP8 KV cache remains an explicit research path instead of a default feature. To use it, pass
`kv_cache_dtype="fp8_e4m3"` and run on a PyTorch / FlashAttention stack where the decode kernel exposes
`k_descale` and `v_descale`. If that interface is missing, nano-vLLM raises a clear error instead of
silently running an incorrect FP8 path.

Future work includes validating FP8 KV cache on a compatible FlashAttention 3 build, FP8 GEMM
microbenchmarks, and TMA experiments for paged KV-cache memory movement.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
