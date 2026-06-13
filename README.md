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
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
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

## Blackwell / RTX 5090 FP8 KV Cache Work

This fork includes an experimental FP8 KV cache path intended for Blackwell-class GPUs such as RTX 5090.
It keeps the normal BF16 path as the default fallback and enables FP8 automatically only when the GPU and
FlashAttention build expose the required FP8 KV-cache descale interface.

```python
from nanovllm import LLM

llm = LLM(
    "/YOUR/MODEL/PATH",
    kv_cache_dtype="auto",      # "bf16", "fp8_e4m3", or "auto"
    kv_cache_scale=1.0,         # static per-layer/per-KV-head scale, v1 default
)
```

What is implemented:

* FP8 E4M3 paged KV cache allocation with `auto` / `bf16` / `fp8_e4m3` configuration.
* Fused Triton BF16-to-FP8 KV store kernel with per-layer, per-KV-head static scales.
* FlashAttention KV-cache decode integration that passes `k_descale` / `v_descale` when available.
* Correctness tests for Triton kernels and opt-in integration tests for BF16 vs FP8 generation.
* Reproducible scripts for KV-cache capacity, decode throughput, FP8 GEMM exploration, and TMA feasibility.

Known v1 boundary: FP8 KV cache currently targets decode. Prefix-cache prefill with reused cached blocks
requires a separate FP8-aware varlen attention path and is intentionally rejected instead of silently
falling back to an incorrect path.

Useful commands:

```bash
pytest tests/test_kernels.py
NANOVLLM_TEST_MODEL=/path/to/Qwen3-0.6B pytest tests/test_fp8_integration.py
python bench_fp8_kvcache.py --model /path/to/Qwen3-0.6B --max-model-len 4096
python bench_fp8_gemm.py
python experiments/tma_kvcache_spike.py
```

For resume reporting, capture the BF16 vs FP8 KV block count, generated tokens/s, peak memory, and
inter-token latency slope across context lengths. The expected headline is that FP8 halves KV-cache
storage per token, so the allocated KV block count should approach 2x when KV cache dominates free memory.


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
