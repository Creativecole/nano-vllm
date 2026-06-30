<p align="center">
<img width="300" src="assets/logo.png">
</p>

# Nano-vLLM Triton Attention Backend

This fork turns nano-vLLM into a focused Qwen3-4B attention-backend project for RTX 5090. It keeps
the original FlashAttention runtime path as the default, and adds an explicit eager-mode
`triton_paged_decode` / `triton_paged_decode_v2` backends for single-token decode with GQA and
paged KV-cache block tables.

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
| E2E backend switch | `bench_e2e.py --attn-backend flash_attn|torch_paged|triton_paged_decode|triton_paged_decode_v2` |
| Profiling | `profile_e2e.py` reports operator attribution and CUDA kernel self-time categories |
| Validation | Tests for GQA mapping, paged KV layout, Qwen3 shapes, attention correctness, and KV store |

## Backend Policy

| Backend | Role |
|---|---|
| `flash_attn` | Default nano-vLLM runtime path |
| `torch_paged` | Correctness reference for paged decode attention |
| `triton_paged_decode` | Custom decode-only Triton PagedAttention v1 backend |
| `triton_paged_decode_v2` | GQA-grouped Triton PagedAttention v2 backend |
| `torch_sdpa` | Prefill benchmark reference |

The Triton paged decode backends currently target BF16/FP16 decode attention with `head_dim=128`,
GQA, online softmax, and paged KV block tables. Runtime use is explicit and eager-only:
`--attn-backend triton_paged_decode_v2 --enforce-eager`.

## RTX 5090 Results

Environment: Qwen3-4B, BF16, single RTX 5090, prompt length 512, 4 prompts, 128 generated tokens,
repeat 3. Full artifacts are under [`results/rtx5090_qwen3_4b`](results/rtx5090_qwen3_4b/).

### End-to-End Decode

| Runtime backend | Decode tokens/s mean | Avg ITL mean | TTFT mean | Peak memory |
|---|---:|---:|---:|---:|
| `flash_attn` | 240.34 | 4.17 ms | 97.9 ms | 27.38 GB |
| `triton_paged_decode` | 210.68 | 4.75 ms | 100.2 ms | 27.38 GB |

The Triton backend is wired into the real generation path and produces stable E2E runs. In this
workload, FlashAttention is still faster end-to-end, which is useful evidence for keeping it as the
default runtime path while using Triton PagedAttention for backend study and targeted profiling.

### Paged Decode Attention Microbenchmark

Representative Qwen3-4B BF16 decode attention numbers from
[`qwen3_attention_summary.md`](results/rtx5090_qwen3_4b/qwen3_attention_summary.md):

| Backend | Batch | Context | Block | p50 latency | Tokens/s | Max abs error |
|---|---:|---:|---:|---:|---:|---:|
| `torch_paged` | 4 | 4096 | 16 | 852.17 ms | 4.69 | 0.0000 |
| `triton_paged_decode` | 4 | 4096 | 16 | 0.259 ms | 15445.86 | 0.0001 |
| `torch_paged` | 8 | 8192 | 256 | 3252.71 ms | 2.46 | 0.0000 |
| `triton_paged_decode` | 8 | 8192 | 256 | 0.531 ms | 15055.51 | 0.0001 |

This benchmark isolates the attention backend. It is not reported as an E2E speedup.

### Profiler Snapshot

The E2E profiler shows that Qwen3-4B decode is still dominated by BF16 Linear/GEMM kernels:

| Backend | Linear/GEMM self CUDA | Attention self CUDA | Notable attention kernel |
|---|---:|---:|---|
| `flash_attn` | 424.18 ms | 30.51 ms | FlashAttention split-KV kernels |
| `triton_paged_decode` | 424.31 ms | 2.86 ms | `_triton_paged_decode_kernel` also appears as a separate Triton kernel in the top-op table |

Profiler traces:
[`profile_flash_attn.md`](results/rtx5090_qwen3_4b/profile_flash_attn.md),
[`profile_triton_paged_decode.md`](results/rtx5090_qwen3_4b/profile_triton_paged_decode.md).

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
    attn_backend="triton_paged_decode_v2",
)
```

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

## Notes

- This is a fork of [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm).
- The default `LLM.generate` path remains FlashAttention unless `attn_backend` is explicitly set.
- Benchmark numbers in this README come from generated result files in `results/rtx5090_qwen3_4b`.
- v2 benchmark/profiler results should be generated before claiming v1 -> v2 speedup.
