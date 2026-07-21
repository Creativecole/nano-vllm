# Qwen3.5 Batched Chunked DeltaNet Prefill

## Root Cause

`ModelRunner.prepare_prefill` already packs active requests into one token tensor, and
`HybridStateManager` already gathers and commits request state with batched
`index_select` / `index_copy_` operations. The remaining batch-scaling problem was in
`Qwen3_5GatedDeltaNet._forward_packed`: packed hidden states were split by request and
the recurrence was called once per sequence per DeltaNet layer.

For four equal-length prompts, each of the 24 DeltaNet layers therefore executed four
batch-1 chunked recurrences instead of one batch-4 recurrence. The math was already
batch-aware; the serving adapter discarded that dimension before calling it.

## Equal-Length Fast Path

For the `chunked` backend, an equal-length packed prefill now reshapes
`[B * T, hidden]` to `[B, T, hidden]` and calls `_forward_stateful_chunk` once. Query,
key, value, convolution state, and FP32 recurrent state keep their batch dimension
through the chunked matrix operations. Output is flattened back to packed order after
the layer finishes.

Single-token decode is unchanged and still uses sequential recurrence. Variable-length
packed prefill retains the previous per-sequence fallback so no padding semantics or
state isolation assumptions are introduced.

The state pool remains request-indexed. For each engine prefill step it performs one
batched gather and one batched commit per DeltaNet layer, using the active request
`slot_ids`; there is no Python copy loop over requests.

## Diagnosis

The diagnostic script records each DeltaNet layer's recurrence calls and Q/K/V/state
shapes, plus state gather/commit calls and payload bytes:

```bash
python benchmarks/qwen35_hybrid/diagnose_batched_prefill.py \
  --model ../models/Qwen3.5-9B \
  --cases 1x128,4x128,4x512 \
  --deltanet-chunk-size 64 \
  --warmup 1 \
  --save-json benchmarks/qwen35_hybrid/results/diagnose_batched_prefill.json \
  --save-md docs/qwen35_hybrid/08_batched_prefill_diagnosis.md
```

For an equal-length batch-4 case, the acceptance signal is one recurrence call per
DeltaNet layer and a recorded query shape whose first dimension is 4.

## Correctness Coverage

Tests compare batched execution with independent per-sequence execution for batch sizes
1, 2, and 4 and prompt lengths 1, 31, 64, 65, 127, 128, 129, and 512. They cover chunk
tails, FP32 recurrent state, BF16-capable output, state isolation, variable-length
fallback, and prefill followed by multiple single-token decode updates.

## Performance Validation

The pre-change RTX 5090 measurements motivating this work were:

| Batch | Prompt | HF TTFT | nano chunked TTFT |
|---:|---:|---:|---:|
| 1 | 128 | 181 ms | 170 ms |
| 1 | 512 | 219 ms | 201 ms |
| 1 | 2048 | 366 ms | 330 ms |
| 4 | 128 | 185 ms | 607 ms |
| 4 | 512 | 241 ms | 739 ms |
| 4 | 2048 | 1001 ms | 1429 ms |

Regenerate the post-change results instead of inferring a speedup from these baselines:

```bash
python benchmarks/qwen35_hybrid/bench_e2e.py \
  --model ../models/Qwen3.5-9B \
  --backends hf,nanovllm_chunked \
  --batch-sizes 1,4 \
  --prompt-lens 128,512,2048 \
  --output-lens 128 \
  --warmup 2 \
  --repeat 5 \
  --deltanet-chunk-size 64 \
  --save-json benchmarks/qwen35_hybrid/results/e2e_batched_chunked.json \
  --save-md docs/qwen35_hybrid/e2e_batched_chunked.md
```

The post-change report should check batch-4 TTFT, batch-1 regression, decode throughput,
peak memory, and state movement counters. No performance improvement is claimed until
that GPU run is complete.
