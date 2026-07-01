<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Triton Attention Backend

This fork turns nano-vLLM into a focused Qwen3-4B attention-backend project for RTX 5090. It keeps
the original FlashAttention runtime path as the default, and adds an explicit eager-mode
`triton_paged_decode` / `triton_paged_decode_v2` / `triton_paged_decode_auto` backends for
single-token decode with GQA and paged KV-cache block tables.

The project is built around one practical workflow:

```text
Qwen3-4B shapes -> paged KV layout -> Torch reference -> Triton decode backend
               -> microbenchmark -> E2E benchmark -> profiler trace
```

## What Is Added

| Area | Added in this fork |
|---|---|
| Qwen3-4B shapes | Reads `hidden_size`, Q heads, KV heads, head dim, GQA ratio, and attention shapes from HF config |
| Paged decode attention | `torch_paged` reference, `triton_paged_decode` v1, and `triton_paged_decode_v2` over block tables |
| GQA support | Maps query heads to KV heads with `kv_head = q_head // GQA_ratio` |
| E2E backend switch | `bench_e2e.py --attn-backend flash_attn|torch_paged|triton_paged_decode|triton_paged_decode_v2|triton_paged_decode_auto` |
| Profiling | `profile_e2e.py` reports operator attribution and CUDA kernel self-time categories |
| Validation | Tests for GQA mapping, paged KV layout, Qwen3 shapes, attention correctness, and KV store |

## Backend Policy

| Backend | Role |
|---|---|
| `flash_attn` | Default nano-vLLM runtime path |
| `torch_paged` | Correctness reference for paged decode attention |
| `triton_paged_decode` | Custom decode-only Triton PagedAttention v1 backend |
| `triton_paged_decode_v2` | GQA-grouped Triton PagedAttention v2 backend |
| `triton_paged_decode_auto` | Experimental threshold policy that picks v1/v2 by context length |
| `torch_sdpa` | Prefill benchmark reference |

The Triton paged decode backends currently target BF16/FP16 decode attention with `head_dim=128`,
GQA, online softmax, and paged KV block tables. Runtime use is explicit and eager-only:
`--attn-backend triton_paged_decode_v2 --enforce-eager`.

## RTX 5090 Results

Environment: Qwen3-4B, BF16, single RTX 5090, prompt length 512, 4 prompts. The E2E table uses
128 generated tokens, block size 256, and repeat 3. Profiler cleanup numbers use 64 profiled
steps. Full artifacts are under [`results/rtx5090_qwen3_4b`](results/rtx5090_qwen3_4b/).

### End-to-End Decode after RoPE Cleanup

| Runtime backend | Decode tokens/s mean | Avg ITL mean | Decode step p50 | Decode step p95 |
|---|---:|---:|---:|---:|
| `flash_attn` | 263.09 | 3.80 ms | 14.81 ms | 15.28 ms |
| `triton_paged_decode` | 247.68 | 4.04 ms | 16.10 ms | 16.52 ms |
| `triton_paged_decode_v2` | 247.71 | 4.04 ms | 16.06 ms | 16.52 ms |

After RoPE indexing cleanup, FlashAttention remains the fastest end-to-end backend. The Triton v1/v2
backends are effectively tied in E2E throughput on this workload. The v2 backend is kept as a
profiler-guided attention kernel iteration because it reduces Triton decode attention kernel
self-time, but the full E2E decode path is still dominated by BF16 Linear/GEMM and runtime overhead.

### Paged Decode Attention Microbenchmark

Representative Qwen3-4B BF16 decode attention numbers from
[`triton_paged_decode_v2.md`](results/rtx5090_qwen3_4b/triton_paged_decode_v2.md):

| Batch | Context | Block | v1 p50 | v2 p50 | v1 -> v2 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 32 | 0.2690 ms | 0.0964 ms | 2.79x faster |
| 4 | 4096 | 16 | 0.2755 ms | 0.1922 ms | 1.43x faster |
| 8 | 8192 | 64 | 0.5481 ms | 0.3729 ms | 1.47x faster |
| 8 | 16384 | 128 | 1.0211 ms | 0.6733 ms | 1.52x faster |

This benchmark isolates the attention backend. It is not reported as an E2E speedup.

### Profiler: Triton PagedAttention v1 -> v2

| Metric | `triton_paged_decode` | `triton_paged_decode_v2` |
|---|---:|---:|
| Triton decode kernel self-time | 81.2 ms | 61.9 ms |
| Triton decode kernel avg latency | 35.8 us/call | 27.3 us/call |
| Attention self CUDA time | 84.0 ms | 64.6 ms |

v2 reduces Triton decode attention kernel self-time by about 24% in this profile run. This is a
kernel/profiler attribution result, not a broad E2E speedup claim.

Profiler traces:
[`profile_flash_attn.md`](results/rtx5090_qwen3_4b/profile_flash_attn.md),
[`profile_triton_paged_decode.md`](results/rtx5090_qwen3_4b/profile_triton_paged_decode.md),
[`profile_triton_paged_decode_v2.md`](results/rtx5090_qwen3_4b/profile_triton_paged_decode_v2.md).

### Profiler-Guided RoPE Indexing Cleanup

The original RoPE path materialized `cos_cache[positions]` and `sin_cache[positions]` through
PyTorch advanced indexing. This created thousands of small indexing and gather operations during
decode.

This fork moves RoPE cos/sin lookup into the Triton rotary kernel. The kernel receives `positions`,
`cos_cache`, and `sin_cache` directly and loads cos/sin using pointer arithmetic. This does not
change attention kernel math.

| Metric | Before cleanup | After cleanup |
|---|---:|---:|
| `aten::index` calls | 4609 | 1 |
| `vectorized_gather_kernel` calls | 4610 | 2 |

The cleanup happens before the attention backend, so `flash_attn`, `triton_paged_decode`, and
`triton_paged_decode_v2` all benefit from it.

### Profiler Interpretation

The profiler shows that Triton decode attention can be optimized locally, but full decode is still
dominated by BF16 Linear/GEMM and runtime overhead. In the same profile setup, Linear/GEMM self CUDA
time is about 421 ms, much larger than the custom attention kernel self-time.

This project reports three separate levels:

1. Microbenchmark: isolated attention backend latency.
2. Profiler: kernel self-time and operator attribution.
3. E2E benchmark: real generation path throughput.

## Install

```bash
git clone https://github.com/Creativecole/nano-vllm.git
cd nano-vllm
pip install -e .
```

Download models:

```bash
hf download Qwen/Qwen3-0.6B --local-dir ../models/Qwen3-0.6B
hf download Qwen/Qwen3-4B --local-dir ../models/Qwen3-4B
```

## Quick Start

Default runtime path:

```python
from nanovllm import LLM, SamplingParams

llm = LLM("../models/Qwen3-4B", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
print(outputs[0]["text"])
```

Triton paged decode runtime path:

```python
llm = LLM(
    "../models/Qwen3-4B",
    enforce_eager=True,
    tensor_parallel_size=1,
    attn_backend="triton_paged_decode",
)
```

`triton_paged_decode_v2` is also available as an experimental profiler-guided kernel iteration, but
it is not the default runtime backend.

## Reproduce

Run tests:

```bash
pytest -q
```

Run the main attention backend benchmark:

```bash
python benchmarks/bench_qwen3_4b_attention.py \
  --model ../models/Qwen3-4B \
  --dtype bf16 \
  --attn-backends torch_paged,triton_paged_decode,triton_paged_decode_v2 \
  --seq-lens 1024,4096,8192 \
  --batch-sizes 1,4,8 \
  --block-size 16 \
  --save-md results/rtx5090_qwen3_4b/qwen3_attention_summary.md \
  --save-json results/rtx5090_qwen3_4b/qwen3_attention_summary.json
```

Run a small correctness spot check with the `torch_paged` reference:

```bash
python benchmarks/bench_triton_paged_decode_v2.py \
  --model ../models/Qwen3-4B \
  --dtype bf16 \
  --backends torch_paged,triton_paged_decode,triton_paged_decode_v2 \
  --seq-lens 1024,4096 \
  --batch-sizes 1,4 \
  --block-sizes 16 \
  --warmup 1 \
  --repeat 3 \
  --save-md results/rtx5090_qwen3_4b/v2_correctness_spotcheck.md \
  --save-json results/rtx5090_qwen3_4b/v2_correctness_spotcheck.json
```

Run the v1/v2 Triton performance sweep without timing the slow reference path:

```bash
python benchmarks/bench_triton_paged_decode_v2.py \
  --model ../models/Qwen3-4B \
  --dtype bf16 \
  --backends triton_paged_decode,triton_paged_decode_v2 \
  --seq-lens 1024,4096,8192,16384 \
  --batch-sizes 1,4,8 \
  --block-sizes 16,32,64,128,256 \
  --warmup 5 \
  --repeat 20 \
  --save-md results/rtx5090_qwen3_4b/triton_paged_decode_v2.md \
  --save-json results/rtx5090_qwen3_4b/triton_paged_decode_v2.json
```

Run E2E benchmarks:

```bash
python bench_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --attn-backend flash_attn \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --save-md results/rtx5090_qwen3_4b/e2e_flash_attn_repeat3.md \
  --save-json results/rtx5090_qwen3_4b/e2e_flash_attn_repeat3.json

python bench_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --attn-backend triton_paged_decode \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --save-md results/rtx5090_qwen3_4b/e2e_triton_paged_decode_repeat3.md \
  --save-json results/rtx5090_qwen3_4b/e2e_triton_paged_decode_repeat3.json

python bench_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --attn-backend triton_paged_decode_v2 \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --save-md results/rtx5090_qwen3_4b/e2e_triton_paged_decode_v2_repeat3.md \
  --save-json results/rtx5090_qwen3_4b/e2e_triton_paged_decode_v2_repeat3.json
```

Run the v1/v2 E2E prompt/block sweep:

```bash
python benchmarks/bench_e2e_v1_v2_sweep.py \
  --model ../models/Qwen3-4B \
  --backends triton_paged_decode,triton_paged_decode_v2 \
  --prompt-lens 512,1024,2048,4096,8192 \
  --block-sizes 16,32,64,128,256 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --enforce-eager \
  --repeat 1 \
  --save-md results/rtx5090_qwen3_4b/e2e_v1_v2_prompt_block_sweep.md \
  --save-json results/rtx5090_qwen3_4b/e2e_v1_v2_prompt_block_sweep.json
```

Compare the auto policy against v1/v2:

```bash
python benchmarks/bench_e2e_attention_compare.py \
  --model ../models/Qwen3-4B \
  --backends triton_paged_decode,triton_paged_decode_v2,triton_paged_decode_auto \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --block-size 256 \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --save-md results/rtx5090_qwen3_4b/e2e_triton_paged_decode_auto.md \
  --save-json results/rtx5090_qwen3_4b/e2e_triton_paged_decode_auto.json
```

Capture a profiler trace:

```bash
python profile_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 64 \
  --attn-backend triton_paged_decode_v2 \
  --enforce-eager \
  --profile-steps 64 \
  --profile-memory \
  --record-shapes \
  --trace-output results/rtx5090_qwen3_4b/profile_triton_paged_decode_v2.json \
  --summary-output results/rtx5090_qwen3_4b/profile_triton_paged_decode_v2.md
```

Generate a v1/v2 profiler diff:

```bash
python profile_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 64 \
  --compare-backends triton_paged_decode,triton_paged_decode_v2 \
  --enforce-eager \
  --profile-steps 64 \
  --profile-memory \
  --record-shapes \
  --trace-output results/rtx5090_qwen3_4b/profile_v1_v2_diff_trace.json \
  --diff-summary-output results/rtx5090_qwen3_4b/profile_v1_v2_diff.md \
  --diff-json-output results/rtx5090_qwen3_4b/profile_v1_v2_diff.json
```

## Notes

- This is a fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
- The default `LLM.generate` path remains FlashAttention unless `attn_backend` is explicitly set.
- Benchmark numbers in this README come from generated result files in `results/rtx5090_qwen3_4b`.
