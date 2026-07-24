# Hybrid Decode Fast Path

## Problem

Qwen3.5 decode executes one token per active request. In a stable batch, the
request order, DeltaNet state slots, attention backend dispatch, and most paged-KV
metadata remain unchanged across many steps. The original path still rebuilt Python
lists, pinned host tensors, CUDA tensors, typed metadata objects, and resident state
views on every step.

This work applies the steady-state preparation idea from
[TensorRT-LLM PR #16313](https://github.com/NVIDIA/TensorRT-LLM/pull/16313)
to nano-vLLM's PyTorch runtime. It does not copy TensorRT-LLM code and does not add
CUDA Graph execution.

## Runtime Flow

Normal decode preparation:

```text
Scheduler
  -> ModelRunner.prepare_decode
     -> rebuild Python metadata
     -> allocate/upload decode tensors
     -> rebuild typed attention metadata
     -> look up resident DeltaNet views
  -> model forward
```

Steady-state decode:

```text
Scheduler
  -> HybridDecodeContext.can_reuse
     -> reuse request/layout mapping
     -> advance positions, context lengths, and slots in place
     -> use previous device-side sampled tokens
     -> reuse typed metadata and resident DeltaNet views
  -> model forward
```

`HybridDecodeContext` retains references and layout identity. Paged KV tensors and
DeltaNet state payloads remain owned by `HybridCacheCoordinator`.

## Reused State

The context caches:

- requested and execution request order;
- resident DeltaNet layer-state views;
- typed `HybridAttentionMetadata`;
- paged-KV block-table tensor and slot mapping;
- context-length, position, and temperature tensors;
- the previous device-side sampled-token tensor;
- the state-slot layout version.

Each reused step updates only positions, context lengths, and KV slots in place. The
sampled token remains on the GPU and becomes the next decode input.

## Eligibility And Invalidation

The fast path is used only when:

- every scheduled request is decoding one token;
- the active request IDs and order are unchanged;
- every sequence advanced by exactly one token;
- DeltaNet state-slot layout is unchanged;
- KV block-table width and final physical block are unchanged;
- sampling temperatures are unchanged;
- the previous sampled-token tensor is available.

It falls back to normal preparation for prefill, mixed prefill/decode, request
arrival or completion, state-slot allocation/compaction, KV block growth, scheduling
mode changes, or sampling changes. The option defaults to `False`; set
`decode_fast_path=True` explicitly for experiments. The original path remains the
default because the first RTX 5090 A/B showed no meaningful E2E improvement.

## Correctness

CPU tests cover stable reuse, in-place buffer identity, mixed-runtime metadata, and
all invalidation boundaries. The CUDA integration test runs the same greedy workload
with the fast path disabled and enabled, then compares generated token IDs and every
captured logits tensor exactly.

Run:

```bash
pytest -q tests/engine/test_decode_context.py \
  tests/engine/test_hybrid_state_manager.py

NANOVLLM_QWEN35_MODEL=../models/Qwen3.5-9B \
pytest -q tests/integration/test_qwen35_generation.py \
  -k decode_fast_path
```

## Benchmark

The A/B benchmark launches normal and fast modes in isolated processes:

```bash
python benchmarks/qwen35_hybrid/bench_decode_fast_path.py \
  --model ../models/Qwen3.5-9B \
  --batch-sizes 1,4,8,16 \
  --prompt-lens 512,2048 \
  --output-len 128 \
  --warmup 1 \
  --repeat 5 \
  --gpu-memory-utilization 0.8 \
  --save-json benchmarks/qwen35_hybrid/results/decode_fast_path_ab.json \
  --save-md docs/qwen35_hybrid/10_decode_fast_path_results.md
```

It reports mean/p50/p95 ITL, decode throughput, peak memory, fast-path hits, builds,
and invalidations. It is an eager BF16 runtime benchmark, not a kernel-speed claim.

Profile one representative case twice:

```bash
python benchmarks/qwen35_hybrid/profile_serving.py \
  --model ../models/Qwen3.5-9B \
  --batch-sizes 4 \
  --prompt-lens 512 \
  --decode-steps 128 \
  --phases decode \
  --deltanet-backend chunked \
  --no-resume \
  --save-json benchmarks/qwen35_hybrid/results/profile_decode_fast.json \
  --save-md docs/qwen35_hybrid/profile_decode_fast.md

python benchmarks/qwen35_hybrid/profile_serving.py \
  --model ../models/Qwen3.5-9B \
  --batch-sizes 4 \
  --prompt-lens 512 \
  --decode-steps 128 \
  --phases decode \
  --deltanet-backend chunked \
  --disable-decode-fast-path \
  --no-resume \
  --save-json benchmarks/qwen35_hybrid/results/profile_decode_normal.json \
  --save-md docs/qwen35_hybrid/profile_decode_normal.md
```

The profiler exposes `qwen35_metadata_prepare` and
`qwen35_decode_prepare_fast` separately. GPU kernel math and latency should remain
unchanged; the intended impact surface is CPU preparation, metadata allocation, H2D
copies, and end-to-end ITL.

## Current Boundary

The fast path is eager-only for Qwen3.5 hybrid execution. It deliberately does not
cover dynamic batch compaction, a step containing new prefill work, CUDA Graph,
multi-GPU tensor parallelism, or prefill/decode disaggregation. Those cases use the
existing normal path.
