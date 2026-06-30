# Paged Attention

Paged attention stores KV cache in fixed-size physical blocks instead of one contiguous tensor per
sequence.

## Decode Shape

For single-token decode:

| tensor | shape |
|---|---|
| `q` | `[batch_size, num_q_heads, head_dim]` |
| `k_cache` | `[num_blocks, block_size, num_kv_heads, head_dim]` |
| `v_cache` | `[num_blocks, block_size, num_kv_heads, head_dim]` |
| `block_tables` | `[batch_size, max_num_blocks_per_seq]` |
| `context_lens` | `[batch_size]` |

## Block Table

Logical token position maps to physical cache storage as:

```text
logical_block = token_position // block_size
block_offset  = token_position % block_size
physical_block = block_tables[sequence_id, logical_block]
cache address = [physical_block, block_offset, kv_head, :]
```

This indirection lets sequences share or move physical KV blocks without rewriting logical sequence
state.

## GQA Mapping

Qwen-style GQA has more query heads than KV heads. The mapping is:

```text
kv_head = q_head // (num_q_heads // num_kv_heads)
```

For example, `num_q_heads=32` and `num_kv_heads=8` means every four query heads share one KV head.

## Online Softmax

Decode attention should not materialize a full attention score matrix. The Triton kernel iterates
over KV chunks and maintains:

- running max
- running denominator
- accumulated weighted V

This is the same numerical idea behind FlashAttention, adapted to paged KV lookup.

## Why Decode Is KV Sensitive

Decode has only one query token per sequence, so GEMM-like QK work is small while K/V reads scale with
context length. Long context decode is therefore sensitive to block-table indirection, memory
coalescing, cache layout, and how much KV data the kernel touches per generated token.

