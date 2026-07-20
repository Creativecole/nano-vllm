# Qwen3.5 Text-Only Model

## Scope

Phase 1 adds a text-only, no-cache Qwen3.5 reference model. It is intended for
configuration validation, strict checkpoint loading, and numerical comparison with
the official Transformers implementation. It does not yet participate in nano-vLLM's
scheduler, paged KV cache, CUDA Graph, or token-by-token serving path.

The implementation reads every model dimension and the layer schedule from the
Hugging Face text config. The observed Qwen3.5-9B config has 32 decoder layers with
24 `linear_attention` layers and 8 `full_attention` layers, but those counts are not
hardcoded. Its observed schedule is `[linear, linear, linear, full]` repeated eight
times, placing full-attention layers at zero-based indices 3, 7, ..., 31.

## Layer Dispatch

`Qwen3_5Model` constructs decoder layers from `text_config.layer_types`. Each entry
must be either:

- `full_attention`: gated GQA with partial RoPE.
- `linear_attention`: Gated DeltaNet.

Every decoder layer uses pre-norm residual ordering:

```text
x = x + token_mixer(RMSNorm(x))
x = x + MLP(RMSNorm(x))
```

Qwen3.5 RMSNorm is one-centered: its learned scale is `(1 + weight)`, with `weight`
initialized to zero.

## Full Attention

For the observed Qwen3.5-9B text config:

- Query heads: 16
- KV heads: 4
- Head dimension: 256
- GQA ratio: 4 query heads per KV head
- Partial RoPE dimension: `head_dim * partial_rotary_factor = 256 * 0.25 = 64`

These values remain config-driven. The Q projection produces both Q and an attention
gate:

```text
q_and_gate = q_proj(x)                         [B, T, Hq, 2 * D]
q, gate = split(q_and_gate)                    [B, T, Hq, D] each
k = k_proj(x)                                  [B, T, Hkv, D]
v = v_proj(x)                                  [B, T, Hkv, D]
q = RMSNorm(q); k = RMSNorm(k)
q, k = partial_rope(q, k)
attention = softmax(q @ k.T / sqrt(D)) @ v
output = o_proj(attention * sigmoid(gate))
```

Only the rotary prefix of Q and K is transformed. V is not rotated. The no-cache
reference repeats K/V heads to the query-head count before eager attention; a future
serving backend can keep the compact GQA representation.

## Gated DeltaNet

The linear-attention token mixer contains these checkpoint-visible projections:

- `in_proj_qkv`: Q, K, and V input to the depthwise causal convolution.
- `in_proj_z`: output gate consumed by gated RMSNorm.
- `in_proj_b`: beta update strength.
- `in_proj_a`: input-dependent decay.
- `out_proj`: maps concatenated value heads back to hidden size.

For the observed Qwen3.5-9B text config, Q/K use 16 heads of dimension 128 and V uses
32 heads of dimension 128. Q/K are repeated from key heads to value heads before the
recurrent rule. The depthwise convolution width is 4.

The clear PyTorch reference accumulates the recurrent state in FP32. Its shape is:

```text
state: [batch, linear_num_value_heads, linear_key_head_dim, linear_value_head_dim]
```

For each token and value head:

```text
q = l2_normalize(q) / sqrt(key_dim)
k = l2_normalize(k)
g = -exp(A_log) * softplus(a + dt_bias)
beta = sigmoid(b)

state = exp(g) * state
retrieved = k @ state
delta = beta * (v - retrieved)
state = state + outer(k, delta)
output = q @ state
```

The output is reshaped per value head, normalized, multiplied by `SiLU(z)`, flattened,
and projected by `out_proj`.

## Prefill And Future Decode

The Phase 1 no-cache path starts the causal convolution and recurrent matrix state at
zero, then processes the complete input sequence. It is deliberately simple and is
not optimized.

Stateful serving requires two different cache families:

- Full-attention layers need paged K/V blocks.
- DeltaNet layers need a width-4 convolution history and an FP32 recurrent matrix per
  active sequence and layer.

Managing those states across scheduling, preemption, prefix reuse, and batch compaction
belongs to Phase 2. `ModelRunner` rejects Qwen3.5 stateful serving in Phase 1 instead of
silently applying the existing pure-attention cache assumptions.
