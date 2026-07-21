# Qwen3.5 Chunked Recurrence Reference

## Why This Backend Exists

The Phase 5 layer profile identifies token-level PyTorch recurrence as the Qwen3.5
prefill launch bottleneck. On Qwen3.5-9B / RTX 5090 / BF16, the sequential DeltaNet
reference produced the following single-layer serving-prefill baseline:

| Prompt | DeltaNet CUDA ms | Kernels | Kernels/token | Full Attention kernels |
|---:|---:|---:|---:|---:|
| 128 | 65.615 | 1,338 | 10.453 | 44 |
| 512 | 251.584 | 5,181 | 10.119 | 44 |
| 2048 | 1006.035 | 20,553 | 10.036 | 44 |

The near-constant `~10 kernels/token` confirms that the explicit sequence-length loop,
not causal convolution, causes the launch explosion.

## Execution Model

`deltanet_backend=sequential` remains the conservative default. The opt-in
`deltanet_backend=chunked` path follows the official Transformers pure-PyTorch chunked
Gated Delta Rule:

- Q/K normalization, decay, beta, and recurrence equations are unchanged.
- Recurrent state accumulation remains FP32.
- Model inputs and recurrence outputs remain BF16 (or FP16 when configured).
- Prefill uses fixed-size chunk matrix operations and one state update per chunk.
- Single-token decode deliberately uses the sequential recurrent update.

This is a reference execution-model experiment, not a fused or Triton kernel.

## Correctness

Run tiny-model unit tests first:

```bash
pytest -q \
  tests/models/test_qwen35_deltanet_reference.py \
  tests/models/test_qwen35_stateful_deltanet.py \
  tests/models/test_qwen35_no_cache_forward.py
```

Then compare a real checkpoint against Hugging Face, including layer logits and greedy
tokens:

```bash
python benchmarks/qwen35_hybrid/validate_correctness.py \
  --model ../models/Qwen3.5-9B \
  --deltanet-backend chunked \
  --deltanet-chunk-size 64 \
  --prompt-lens 1,16,128,512 \
  --batch-sizes 1,2,4 \
  --decode-steps 1,8,32 \
  --save-json benchmarks/qwen35_hybrid/results/chunked_correctness.json \
  --save-md docs/qwen35_hybrid/06_chunked_correctness.md
```

## Layer Profile: Kernel Count, CUDA Time, Memory

Use a new output path so the original Phase 5 baseline remains intact:

```bash
python benchmarks/qwen35_hybrid/profile_layers.py \
  --model ../models/Qwen3.5-9B \
  --layer-id 0 \
  --batch-size 1 \
  --prompt-len 128 \
  --prompt-len 512 \
  --prompt-len 2048 \
  --deltanet-backends sequential,chunked \
  --deltanet-chunk-size 64 \
  --warmup 1 \
  --repeat 3 \
  --profile-memory \
  --save-json benchmarks/qwen35_hybrid/results/deltanet_backend_profile.json \
  --save-md docs/qwen35_hybrid/06_chunked_recurrence_profile.md
```

The generated JSON reports CUDA time, kernel count, kernels/input-token, peak allocated
memory, and peak temporary-memory delta. It also computes sequential-to-chunked CUDA
speedup and kernel-count reduction without inventing results.

## E2E: TTFT and Memory

```bash
python benchmarks/qwen35_hybrid/bench_e2e.py \
  --model ../models/Qwen3.5-9B \
  --backends nanovllm_sequential,nanovllm_chunked \
  --batch-sizes 1 \
  --prompt-lens 128,512,2048 \
  --output-lens 32 \
  --deltanet-chunk-size 64 \
  --warmup 1 \
  --repeat 5 \
  --save-json benchmarks/qwen35_hybrid/results/chunked_e2e.json \
  --save-md docs/qwen35_hybrid/06_chunked_e2e.md
```

TTFT is the primary expected benefit because the new backend changes prefill. Decode
throughput is retained as a regression check, and peak memory makes the chunked
algorithm's temporary-storage tradeoff visible.

The completed RTX 5090 results and the Nsight Systems follow-up are summarized in
`07_chunked_recurrence_final.md`.
