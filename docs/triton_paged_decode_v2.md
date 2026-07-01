# Triton Paged Decode v2

This note tracks the second optimization iteration for the decode-only Triton PagedAttention backend.
The target workload is Qwen3-4B on RTX 5090: 32 query heads, 8 KV heads, `head_dim=128`, BF16, and
paged KV cache.

## v1 Baseline

`triton_paged_decode` maps one Triton program to one `(batch, q_head)` pair. Each program walks the
full context in `BLOCK_N` chunks, reads the block table inside the kernel, loads K/V from
`[num_blocks, block_size, num_kv_heads, head_dim]`, and maintains online softmax state for one query
head.

For Qwen3-4B GQA, four query heads share one KV head. v1 loads the same KV rows separately for each
of those four query heads.

## v2 Change

`triton_paged_decode_v2` maps one Triton program to one `(batch, kv_head)` pair. The program computes
all query heads belonging to that KV head in one pass:

```text
program_id_0: batch
program_id_1: kv_head
inside program: q_heads = kv_head * GQA_GROUP + [0 .. GQA_GROUP)
```

For Qwen3-4B, `GQA_GROUP=4`, so v2 reduces the number of attention programs and reuses K/V loads
across four query heads. It keeps the same paged KV layout and online softmax semantics as v1.

## Benchmark Methodology

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

Run E2E with v2:

```bash
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

Run profiler with v2:

```bash
python profile_e2e.py \
  --model ../models/Qwen3-4B \
  --prompt-len 512 \
  --num-prompts 4 \
  --max-tokens 64 \
  --attn-backend triton_paged_decode_v2 \
  --enforce-eager \
  --profile-memory \
  --record-shapes \
  --trace-output results/rtx5090_qwen3_4b/profile_triton_paged_decode_v2.json \
  --summary-output results/rtx5090_qwen3_4b/profile_triton_paged_decode_v2.md
```

Diagnose why kernel-level gains do or do not survive E2E:

```bash
python benchmarks/bench_e2e_v1_v2_sweep.py \
  --model ../models/Qwen3-4B \
  --backends triton_paged_decode,triton_paged_decode_v2 \
  --prompt-lens 512,1024,2048,4096,8192 \
  --block-sizes 16,32,64,128,256 \
  --num-prompts 4 \
  --max-new-tokens 128 \
  --enforce-eager \
  --save-md results/rtx5090_qwen3_4b/e2e_v1_v2_prompt_block_sweep.md \
  --save-json results/rtx5090_qwen3_4b/e2e_v1_v2_prompt_block_sweep.json

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

## Current Status

- v1 remains available as `triton_paged_decode`.
- v2 is available as `triton_paged_decode_v2`.
- `triton_paged_decode_auto` is available as an experimental threshold policy.
- The default runtime backend remains `flash_attn`.
- v2 is decode-only and eager-mode only.
- Split-KV parallelism is not enabled in the main path; it remains a separate future prototype.
