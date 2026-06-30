<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM Triton Attention Backend

This fork is being shaped into a focused attention-kernel project for Qwen3-4B on RTX 5090. The main
track is decode-only Triton PagedAttention with GQA and paged KV-cache block tables, benchmarked
against Torch reference paths and FlashAttention where available.

The default generation path still uses nano-vLLM's existing FlashAttention integration. The custom
Triton PagedAttention backend is available as an explicit eager-mode decode backend, so correctness,
latency, profiling, and E2E behavior can be evaluated without changing the stable default path.

## Project Scope

| Area | Current status |
|---|---|
| Qwen3-4B shapes | Loaded from HuggingFace config; no hard-coded head counts or hidden sizes |
| Paged KV cache | Explicit block table layout for decode attention benchmarks |
| GQA | Query-head to KV-head mapping tested and used by Torch/Triton decode paths |
| Decode backend | `torch_paged` reference and `triton_paged_decode` custom kernel |
| Prefill benchmark | Torch SDPA and FlashAttention comparison; Triton flash-style prefill remains TODO |
| E2E generation | Stable `flash_attn` path preserved; `triton_paged_decode` can be enabled explicitly in eager mode |

## Attention Backend Model

```text
Qwen3 config
   |
   v
Q/K/V shapes + GQA ratio
   |
   v
Paged KV cache: [num_blocks, block_size, num_kv_heads, head_dim]
   |
   v
block_tables: [batch, max_blocks_per_seq]
   |
   v
single-token decode q: [batch, num_q_heads, head_dim]
   |
   v
torch_paged reference  <->  triton_paged_decode backend
```

Supported backend names:

| Backend | Use |
|---|---|
| `torch_sdpa` | Prefill reference path using PyTorch SDPA |
| `flash_attn` | Stable nano-vLLM runtime attention and optional prefill benchmark path |
| `torch_paged` | Decode correctness reference over paged KV cache |
| `triton_paged_decode` | Decode-only Triton PagedAttention kernel |

The Triton decode kernel currently targets single-token decode, GQA, paged KV block tables,
`fp16`/`bf16`, and `head_dim=128`. It uses online softmax and does not materialize the full attention
score matrix.

Runtime note: `torch_paged` and `triton_paged_decode` currently require `--enforce-eager`. CUDA Graph
capture for custom decode backends is intentionally left as a later integration step.

## Main Commands

Decode-only attention backend benchmark:

```bash
python benchmarks/bench_attention_decode.py \
  --model Qwen/Qwen3-4B \
  --dtype bf16 \
  --backend triton_paged_decode \
  --seq-lens 1024,4096 \
  --batch-sizes 1,4 \
  --block-sizes 16 \
  --save-md results/rtx5090_qwen3_4b/attention_decode.md \
  --save-json results/rtx5090_qwen3_4b/attention_decode.json
```

Qwen3-4B attention summary:

```bash
python benchmarks/bench_qwen3_4b_attention.py \
  --model Qwen/Qwen3-4B \
  --dtype bf16 \
  --attn-backends torch_paged,triton_paged_decode \
  --seq-lens 1024,4096,8192 \
  --batch-sizes 1,4,8 \
  --block-size 16 \
  --save-md results/rtx5090_qwen3_4b/qwen3_attention_summary.md \
  --save-json results/rtx5090_qwen3_4b/qwen3_attention_summary.json
```

Prefill attention comparison:

```bash
python benchmarks/bench_attention_prefill.py \
  --model Qwen/Qwen3-4B \
  --dtype bf16 \
  --backends torch_sdpa,flash_attn \
  --seq-lens 512,1024 \
  --batch-sizes 1,4 \
  --save-md results/rtx5090_qwen3_4b/attention_prefill.md \
  --save-json results/rtx5090_qwen3_4b/attention_prefill.json
```

No benchmark numbers are claimed until the scripts are run on the target RTX 5090 environment.

End-to-end Triton paged decode experiment:

```bash
python bench_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --attn-backend triton_paged_decode \
  --enforce-eager \
  --save-md results/rtx5090_qwen3_4b/e2e_triton_paged_decode.md \
  --save-json results/rtx5090_qwen3_4b/e2e_triton_paged_decode.json
```

## Profiling Direction

The next profiling target is the decode attention path itself:

- block-table indirect reads from paged KV cache
- GQA query-head to KV-head mapping
- context-length masking
- online softmax state updates
- KV memory traffic as context length grows

The goal is to compare `torch_paged`, `triton_paged_decode`, and the stable FlashAttention runtime
path on the same Qwen3-4B shapes, then use profiler evidence to decide whether the custom backend is
worth further optimizing and extending to CUDA Graph replay.

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

Download an example model:

```bash
hf download Qwen/Qwen3-0.6B --local-dir ../models/Qwen3-0.6B
hf download Qwen/Qwen3-4B --local-dir ../models/Qwen3-4B
```

Hybrid checkpoints with `linear_attn` weights need a separate model adapter and are intentionally
rejected by this fork.

## Quick Start

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "/YOUR/MODEL/PATH",
    enforce_eager=True,
    tensor_parallel_size=1,
)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
outputs = llm.generate(["Hello, Nano-vLLM."], sampling_params)
outputs[0]["text"]
```

## How To Reproduce

Run CPU-safe and GPU kernel tests:

```bash
pytest tests/test_block_manager.py
pytest tests/test_gqa_head_mapping.py tests/test_paged_kv_layout.py tests/test_qwen3_shapes.py
pytest tests/test_attention_correctness.py
pytest tests/test_kv_cache_store.py
NANOVLLM_TEST_MODEL=../models/Qwen3-0.6B pytest tests/test_integration.py -q
```

Run the new attention backend benchmarks:

```bash
python benchmarks/bench_attention_decode.py \
  --model ../models/Qwen3-4B \
  --dtype bf16 \
  --backend triton_paged_decode \
  --seq-lens 1024,4096 \
  --batch-sizes 1,4 \
  --block-sizes 16 \
  --save-md results/rtx5090_qwen3_4b/attention_decode.md \
  --save-json results/rtx5090_qwen3_4b/attention_decode.json

python benchmarks/bench_qwen3_4b_attention.py \
  --model ../models/Qwen3-4B \
  --dtype bf16 \
  --attn-backends torch_paged,triton_paged_decode \
  --seq-lens 1024,4096,8192 \
  --batch-sizes 1,4,8 \
  --block-size 16 \
  --save-md results/rtx5090_qwen3_4b/qwen3_attention_summary.md \
  --save-json results/rtx5090_qwen3_4b/qwen3_attention_summary.json
```

Run end-to-end generation benchmark for the stable FlashAttention path:

```bash
python bench_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --enforce-eager \
  --warmup 1 \
  --repeat 3 \
  --save-md results/rtx5090_qwen3_4b/e2e_qwen3_4b_5090_repeat3.md \
  --save-json results/rtx5090_qwen3_4b/e2e_qwen3_4b_5090_repeat3.json
```

Run end-to-end generation benchmark with Triton paged decode enabled:

```bash
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
```

Capture a PyTorch profiler trace:

```bash
python profile_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 128 \
  --attn-backend triton_paged_decode \
  --enforce-eager \
  --profile-steps 64 \
  --profile-memory \
  --record-shapes \
  --trace-output results/rtx5090_qwen3_4b/profile_qwen3_4b_5090.json \
  --summary-output results/rtx5090_qwen3_4b/profile_qwen3_4b_5090.md
```

## Script Map

| Script | Role |
|---|---|
| `benchmarks/bench_attention_decode.py` | Decode-only paged attention backend benchmark |
| `benchmarks/bench_attention_prefill.py` | Torch SDPA / FlashAttention prefill benchmark |
| `benchmarks/bench_qwen3_4b_attention.py` | Main Qwen3-4B attention backend summary |
| `bench_e2e.py` | E2E baseline and explicit eager custom decode backend runs |
| `profile_e2e.py` | PyTorch profiler trace and top-op summary for runtime bottlenecks |

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
