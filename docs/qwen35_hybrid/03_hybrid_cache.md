# Qwen3.5 Hybrid Cache Execution

## State Types

Phase 2 separates layer state by the configured `layer_types` schedule:

- `PagedKVState` belongs to a full-attention layer and owns global physical K/V block
  tensors. Per-request block tables remain owned by the existing `BlockManager`.
- `DeltaNetState` belongs to a linear-attention layer and is allocated from a fixed
  request slot pool.

For each request and DeltaNet layer:

```text
conv_state:
  [linear_key_dim * 2 + linear_value_dim, linear_conv_kernel_dim]
  model dtype (BF16 for Qwen3.5-9B)

recurrent_state:
  [linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]
  FP32
```

The observed Qwen3.5-9B shape is `[8192, 4]` for convolution state and
`[32, 128, 128]` for recurrent state, but allocation is derived from config rather
than these numbers.

## Lifecycle

1. Scheduler admits a waiting request using existing paged-KV block capacity.
2. ModelRunner allocates one hybrid state slot keyed by `seq_id` before its first
   prefill execution.
3. Packed prefill gathers request rows, updates convolution and recurrent states, and
   commits them back to their slots.
4. Decode gathers rows in the current active-batch order. This makes batch reorder and
   compaction independent of physical slot order.
5. Completion or preemption emits a release event. ModelRunner zeroes and returns the
   slot immediately. A preempted request recomputes from a zero state on readmission.

Shared prefix reuse is disabled for the hybrid path because a reused full-attention KV
prefix would also require a matching DeltaNet state snapshot. Chunked prefill for the
same request remains valid: its own prior KV blocks and DeltaNet states are retained.

## Prefill And Decode

Prefill keeps nano-vLLM's packed token representation. Full-attention layers execute
the existing FlashAttention varlen/paged path. DeltaNet layers split the packed hidden
states by `prefill_seq_lens`, update each request state independently, and concatenate
the outputs back into packed order.

Decode receives one token per active request. Full-attention layers store K/V through
the existing slot mapping and read through the request block table. DeltaNet layers
run one recurrent update from the gathered request state.

CUDA Graph, Triton kernels, prefix-cache snapshots, and performance tuning are outside
Phase 2. Qwen3.5 hybrid serving is forced to eager execution.

## Memory Accounting

KV token capacity continues to use only the sum of full-attention layer block sizes.
DeltaNet state bytes are calculated separately from every `DeltaNetStateSpec`. Automatic
state capacity reserves `hybrid_state_memory_fraction` of the available model-runner
budget; `hybrid_state_capacity` can set an explicit slot count for correctness runs.
