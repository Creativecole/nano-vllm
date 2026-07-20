# Qwen3.5 Hybrid Serving: Repository Analysis

## 1. Baseline provenance

This worktree was created from the unmodified upstream nano-vLLM repository before
any Qwen3.5 work started.

| Item | Value |
|---|---|
| Upstream repository | `https://github.com/GeeeekExplorer/nano-vllm.git` |
| Clean baseline | `bb823b3e06983d71485a8e1f23715ebd87d98ef8` |
| Baseline subject | `Merge pull request #218 from GeeeekExplorer/chunked-prefill-refactor` |
| Feature branch | `feature/qwen35-hybrid-serving` |
| Isolated worktree | `/Users/nonakaharu/Desktop/nano-vllm-qwen35` |
| Baseline diff before edits | empty |

The existing `Creativecole/nano-vllm` worktree remains on its original `main`
branch. Its untracked benchmark artifacts and interview notes were not moved,
deleted, stashed, or modified.

## 2. Analysis sources

The upstream nano-vLLM source is the implementation baseline. Qwen3.5 model facts
were checked against the following sources on 2026-07-21 instead of inferred from
Qwen3:

- `Qwen/Qwen3.5-9B` `config.json` and `model.safetensors.index.json`.
- Hugging Face Transformers Qwen3.5 configuration and model implementation at
  Transformers commit `cbf4d720ec734edb77d25452b2790c7e4be2f8d7`.
- A local vLLM checkout at commit
  `8a83e6f2d752660dfe598395fb13809bb55fb53f`, used as an implementation reference
  for hybrid cache and Gated DeltaNet state handling. It is not a source to copy
  wholesale.

Relevant primary files:

- `transformers/models/qwen3_5/configuration_qwen3_5.py`
- `transformers/models/qwen3_5/modeling_qwen3_5.py`
- `transformers/cache_utils.py`
- `Qwen/Qwen3.5-9B/config.json`
- `Qwen/Qwen3.5-9B/model.safetensors.index.json`

The local Python environment in this worktree does not currently have
Transformers installed. Model-level correctness therefore belongs to a later GPU
environment validation phase.

## 3. Current nano-vLLM execution path

```text
LLM.generate
  -> LLMEngine.add_request
  -> Scheduler.add
  -> loop while requests remain
       -> LLMEngine.step
       -> Scheduler.schedule
            -> BlockManager.can_allocate / allocate     (prefill)
            -> BlockManager.can_append / may_append     (decode)
       -> ModelRunner.run
            -> prepare_prefill or prepare_decode
            -> set global forward Context
            -> model forward
                 -> embedding
                 -> every Qwen3 decoder layer
                      -> RMSNorm
                      -> QKV projection + RoPE
                      -> KV cache store
                      -> FlashAttention prefill/decode
                      -> output projection
                      -> MLP
                 -> final RMSNorm + LM head
            -> Sampler
       -> Scheduler.postprocess
            -> update prefix hashes
            -> append sampled token
            -> deallocate completed requests
```

### 3.1 Key files and responsibilities

| File | Current responsibility | Important assumption |
|---|---|---|
| `nanovllm/llm.py` | Public `LLM` class | Merely subclasses `LLMEngine` |
| `nanovllm/config.py` | Loads `AutoConfig`, validates engine limits | Model dimensions exist directly on top-level config |
| `nanovllm/engine/llm_engine.py` | Request loop, tokenizer, runner and scheduler ownership | Runner mutates cache capacity before scheduler construction |
| `nanovllm/engine/scheduler.py` | Prefill/decode selection, preemption, request completion | Admission pressure is represented only by free KV blocks |
| `nanovllm/engine/block_manager.py` | Physical block allocation, refcounts, prefix hashes | Every cacheable token consumes the same block resource |
| `nanovllm/engine/sequence.py` | Per-request tokens, status and block table | Per-request state is only a token list plus one KV block table |
| `nanovllm/engine/model_runner.py` | Model construction, input metadata, KV allocation, CUDA Graph | Model is hard-coded to Qwen3; every layer receives K/V cache |
| `nanovllm/models/qwen3.py` | The only model implementation | Every decoder layer contains standard self-attention |
| `nanovllm/layers/attention.py` | KV store and FlashAttention calls | Attention module always owns `k_cache` and `v_cache` |
| `nanovllm/utils/context.py` | Process-global forward metadata | Metadata describes only paged attention |
| `nanovllm/utils/loader.py` | Safetensors weight loading and packed projections | No complete loaded/missing/duplicate parameter audit |

The clean baseline has no `tests/` directory and no model registry. Qwen3 support is
not registered dynamically: `ModelRunner` imports and constructs
`Qwen3ForCausalLM` directly.

## 4. Model configuration and loading today

`Config.__post_init__` calls `AutoConfig.from_pretrained(model)` and immediately
reads `hf_config.max_position_embeddings`. `ModelRunner` then reads dimensions such
as `num_hidden_layers`, `num_key_value_heads`, and `dtype` from the same top-level
object.

This works for a plain `Qwen3Config`. It does not work for the Qwen3.5 checkpoint,
whose top-level config is `Qwen3_5Config` and whose language dimensions live in
`hf_config.text_config`. The model loader also cannot currently ignore the vision
tower safely while requiring all text weights to load.

Current Qwen3 weight loading supports these packed mappings:

```text
q_proj, k_proj, v_proj -> qkv_proj
gate_proj, up_proj     -> gate_up_proj
```

It iterates safetensors keys and uses `model.get_parameter`. It does not maintain a
set of expected parameters, loaded parameters, or duplicate writes. The Qwen3.5
loader must add strict accounting rather than silently accepting skipped text
weights.

## 5. Prefill and decode today

### 5.1 Prefill

`Scheduler.schedule` takes requests from `waiting`, checks prefix blocks, allocates
their block tables, and packs scheduled prompt tokens up to
`max_num_batched_tokens`. Only the first request in a scheduling round may use
chunked prefill.

`ModelRunner.prepare_prefill` flattens the scheduled token ranges and builds:

- `input_ids` and `positions`;
- `cu_seqlens_q` and `cu_seqlens_k`;
- `slot_mapping` for physical KV writes;
- a padded `block_tables` tensor only when cached K/V precedes the new query.

The model executes FlashAttention varlen. Every layer writes K/V for the scheduled
tokens through the same slot mapping.

### 5.2 Decode

`Scheduler.schedule` assigns one token per running request. It appends a new
physical block when the next token crosses a block boundary. The selected sequence
list defines batch row order.

`ModelRunner.prepare_decode` builds one row per selected sequence:

- last input token;
- absolute position;
- physical KV slot;
- context length;
- padded block table.

Every attention layer then calls `flash_attn_with_kvcache`. There is no recurrent
state lookup, request-state slot, or state gather/scatter step.

## 6. Current paged KV cache

### 6.1 Physical layout

The runner allocates one uniform tensor:

```text
kv_cache.shape = [
    2,                    # K and V
    num_hidden_layers,
    num_kvcache_blocks,
    block_size,
    num_kv_heads_per_tp_rank,
    head_dim,
]
```

The tensor uses the current default model dtype because no explicit dtype is passed.
Each `Attention` module receives one layer slice by scanning modules that expose
both `k_cache` and `v_cache`.

The block-byte calculation is:

```text
2 * num_hidden_layers * block_size * num_kv_heads * head_dim * dtype_size
```

This assumes every layer has identical paged K/V state.

### 6.2 Logical-to-physical mapping

Each `Sequence` owns `block_table: list[int]`. A logical token position maps to:

```text
logical_block = position // block_size
block_offset  = position % block_size
physical_slot = block_table[logical_block] * block_size + block_offset
```

The Triton KV store kernel writes flattened K/V at `physical_slot`. FlashAttention
receives the block table and context lengths to read the paged cache.

### 6.3 Lifetime

`BlockManager` owns all physical blocks. A block moves between `free_block_ids` and
`used_block_ids`; `ref_count` permits prefix blocks to be shared. Completing or
preempting a request decrements every referenced block and clears the request block
table.

The scheduler is currently the owner of allocation and free events. The runner owns
the physical GPU tensor, while the scheduler owns the block IDs that index it.

## 7. Scheduler and batching assumptions

1. Capacity is one scalar: the number of free KV blocks.
2. Request memory grows with token count only.
3. A new request has no fixed GPU state allocation.
4. Preemption can recover all request cache memory by dropping its block table and
   recomputing prefill later.
5. Every model layer consumes the same block IDs and block size.
6. A `Sequence` object is sufficient to serialize all model state sent to tensor
   parallel workers.
7. Batch order is implicit: the order of `seqs` returned by `schedule` is reused by
   input metadata construction, sampling, and `postprocess` zipping.

For recurrent layers, batch order cannot become state ownership. A request must own
a stable recurrent-state slot, and each batch row must map to that slot. Otherwise,
compaction or sequence completion can silently attach one request's DeltaNet state
to another request.

## 8. Prefix cache today

The block manager hashes complete token blocks using a chained hash. It deliberately
does not match the final incomplete prompt block. Cached blocks retain token/hash
metadata after their refcount reaches zero and can be reactivated from the free
list if the hash and tokens match.

When a prefix hit exists, prefill computes only uncached query tokens and passes the
cached block table to FlashAttention. This is correct for a model whose complete
history is represented by paged K/V.

It is not correct for Qwen3.5 Hybrid. Reusing full-attention K/V without restoring
the DeltaNet convolution and recurrent states produces a model state that never
existed. Phase 1 hybrid serving must therefore disable prefix reuse for Qwen3.5,
while leaving Qwen3 prefix caching unchanged. Hybrid prefix snapshots are a later,
explicit feature.

## 9. CUDA Graph assumptions

The baseline captures decode graphs for batch-size buckets. Captured inputs,
positions, slot mappings, context lengths, block tables, outputs, model parameters,
and cache tensors have stable addresses. Before replay, current metadata is copied
into persistent buffers.

A hybrid graph would additionally require:

- stable recurrent-state pool addresses;
- stable state-index metadata buffers;
- no Python allocation or request-keyed tensor construction inside layer forward;
- inactive padded rows that cannot update a live request's state;
- deterministic in-place recurrent updates under graph replay.

The first Qwen3.5 correctness implementation should run eager. CUDA Graph support
must be enabled only after state ownership and eager outputs are validated.

## 10. Actual Qwen3.5-9B text architecture

The public `Qwen/Qwen3.5-9B` checkpoint is wrapped as
`Qwen3_5ForConditionalGeneration`, but this project intentionally loads only the
language model.

### 10.1 Text configuration

| Field | Qwen3.5-9B value |
|---|---:|
| Hidden size | 4096 |
| Layers | 32 |
| Full-attention interval | 4 |
| Linear-attention layers | 24 |
| Full-attention layers | 8 |
| Attention query heads | 16 |
| Attention KV heads | 4 |
| Full-attention head dim | 256 |
| MLP intermediate size | 12288 |
| Linear key heads | 16 |
| Linear value heads | 32 |
| Linear key/value head dim | 128 / 128 |
| Linear convolution width | 4 |
| Maximum positions | 262144 |
| Model dtype | BF16 |
| Recurrent/SSM dtype in config | FP32 |

The default layer pattern is:

```text
linear_attention, linear_attention, linear_attention, full_attention
```

repeated eight times. The pattern must come from `text_config.layer_types`; it must
not be hard-coded from the interval when the checkpoint provides an explicit list.

### 10.2 Full Attention differences from Qwen3

Qwen3.5 Full Attention is not a drop-in use of `Qwen3Attention`:

- `q_proj` produces both query and per-query-head output gate values;
- Q and K use Qwen3.5 RMSNorm, whose learned scale is `1 + weight`;
- only a fraction of each 256-dimensional head is rotary (`partial_rotary_factor`
  is 0.25 for this checkpoint);
- text RoPE config is under `rope_parameters`, including the Qwen3.5 interleaved
  mRoPE representation;
- attention output is multiplied by `sigmoid(gate)` before `o_proj`;
- only Full Attention layers own paged K/V state.

The first implementation can reuse nano-vLLM's FlashAttention execution and paged
layout, but must implement these surrounding model semantics exactly.

### 10.3 Gated DeltaNet state and recurrence

Each linear-attention layer projects:

```text
Q, K, V       <- in_proj_qkv(hidden)
z             <- in_proj_z(hidden)
beta          <- sigmoid(in_proj_b(hidden))
g             <- -exp(A_log) * softplus(in_proj_a(hidden) + dt_bias)
```

It applies a depthwise causal convolution to Q/K/V, L2-normalizes Q and K, and
updates a recurrent matrix. A simplified single-token recurrence is:

```text
S <- exp(g) * S
retrieved <- K @ S
delta <- beta * (V - retrieved)
S <- S + outer(K, delta)
O <- Q @ S
```

The output is gated RMSNorm of `O` using `z`, followed by `out_proj`.

The request-owned state consists of at least:

1. convolution history for projected Q/K/V;
2. recurrent DeltaNet matrix state.

For the Qwen3.5-9B text config, the reference shapes per linear layer are:

```text
conv_dim       = 2 * 16 * 128 + 32 * 128 = 8192
conv_state     = [8192, 4]                # HF reference cache convention
recurrent_state= [32, 128, 128]
```

The recurrent matrix is FP32 in the reference path. The convolution state follows
the model/cache dtype. Production kernels may store only `kernel_width - 1` history
elements; that is a backend-specific layout and must not be substituted into the
reference path without a direct equivalence test.

Approximate fixed state per request using BF16 convolution state and FP32 recurrent
state:

```text
per linear layer = 8192 * 4 * 2 + 32 * 128 * 128 * 4
                 = 2.0625 MiB
24 linear layers = 49.5 MiB per active request
```

This fixed cost makes `max_num_seqs=512` infeasible if implemented as an
unconditional full-size state pool. State capacity and KV capacity need a joint
memory budget.

### 10.4 Full Attention KV memory

Only eight layers store K/V. In BF16 on one GPU:

```text
bytes per cached token
  = 2 * 8 layers * 4 KV heads * 256 head_dim * 2 bytes
  = 32768 bytes = 32 KiB
```

At a 4096-token context, Full Attention KV is approximately 128 MiB per request.
The unmodified allocator would incorrectly reserve cache for all 32 layers, using
four times the required Full Attention KV storage.

### 10.5 Checkpoint namespaces

The 9B checkpoint contains text weights under `model.language_model.*`, a vision
tower under `model.visual.*`, an `lm_head.weight`, and optional `mtp.*` weights.
Examples include:

```text
model.language_model.embed_tokens.weight
model.language_model.layers.0.linear_attn.in_proj_qkv.weight
model.language_model.layers.3.self_attn.q_proj.weight
model.language_model.layers.3.self_attn.k_proj.weight
model.language_model.layers.3.self_attn.v_proj.weight
model.language_model.layers.31.mlp.down_proj.weight
model.language_model.norm.weight
lm_head.weight
```

Text-only loading needs an explicit namespace mapper and an explicit allowlist for
skipping `model.visual.*` and `mtp.*`. Every required text parameter must be loaded
exactly once.

## 11. Assumptions that Qwen3.5 breaks

| Baseline assumption | Why it fails | Required direction |
|---|---|---|
| One hard-coded Qwen3 model | Qwen3.5 config and layers differ | Minimal model registry/factory |
| Model fields are top-level | Qwen3.5 uses `text_config` | Normalize `hf_text_config` |
| Every layer is attention | 24/32 layers are Gated DeltaNet | Construct layers from `layer_types` |
| Every layer has K/V | DeltaNet has fixed recurrent state | Per-layer state specs |
| One uniform cache tensor | KV and recurrent shapes/lifetimes differ | Hybrid state manager |
| Capacity equals KV blocks | Recurrent state has fixed request cost | Joint admission accounting |
| Prefix K/V is complete state | DeltaNet state is missing | Disable hybrid prefix reuse first |
| Dropping blocks resets a request | Recurrent state also needs reset/free | Atomic state lifecycle |
| Batch row can imply state row | Batch compaction changes row order | Stable request state slot + indices |
| Qwen3 RoPE/RMSNorm applies | Qwen3.5 uses partial mRoPE and `1+w` norm | Dedicated model layers |
| Existing loader is strict enough | Wrapped text/vision/MTP namespaces exist | Audited loader |
| CUDA Graph only needs attention metadata | Recurrent state is mutated in-place | Eager first; static graph metadata later |

## 12. Proposed state architecture

The design should add one ownership boundary rather than scatter model-specific
branches across `ModelRunner`.

```text
LayerStateSpec
  - layer_id
  - kind
  - bytes_per_token(request)
  - fixed_bytes_per_request
  - allocate_physical_storage(...)

PagedKVLayerStateSpec
  - num_kv_heads
  - head_dim
  - block_size
  - dtype

RecurrentLayerStateSpec
  - conv_shape / conv_dtype
  - recurrent_shape / recurrent_dtype

HybridStateManager
  - allocate(request_id)
  - free(request_id)
  - reset(request_id)
  - prepare_batch(request_ids)
  - reorder(old_order, new_order)
  - get_layer_state(layer_id, request_id)
  - memory_usage(request_id)
  - memory_usage_total()
```

### 12.1 Ownership

- `BlockManager` remains the owner of paged Full Attention block IDs and prefix
  hash metadata.
- `HybridStateManager` owns recurrent-state slots and their GPU storage.
- `Sequence` stores a stable recurrent-state slot ID, not recurrent tensors.
- `Scheduler` performs admission/free transactions across both managers.
- `ModelRunner` receives batch metadata and injects physical state storage into
  layers; it does not decide request ownership.
- Model layers read state through a typed forward context and never search by Python
  request ID inside the kernel path.

### 12.2 Request slot mapping

Use a stable pool slot per active request. `prepare_batch(request_ids)` creates a
small state-index tensor in current batch order. Linear layers index their state pool
with that tensor. Batch reorder changes only the index metadata; it does not copy or
reassign the owning recurrent state.

This supports continuous batching and avoids an O(number of layers * state size)
physical reorder each step. A debug/reference path may gather state tensors, but the
serving path should eventually use direct indexed updates.

### 12.3 Lifecycle transaction

| Event | Paged KV action | Recurrent action |
|---|---|---|
| New request admitted | Allocate initial block table | Reserve and zero one state slot |
| Prefill | Write Full Attention K/V | Build conv/recurrent state |
| Decode | Append slot/block as needed | Read and update same request slot |
| Batch reorder | Reorder block-table rows | Reorder state-index metadata only |
| Completion | Decrement/free all blocks | Zero/release state slot |
| Preemption in v1 | Drop K/V and recompute | Reset state and recompute prefill |
| Request reset | Clear block ownership | Zero both state types |

Allocation must be atomic from the scheduler's point of view. If either KV blocks or
a recurrent slot is unavailable, neither resource should remain partially assigned.

## 13. Scheduler memory accounting

The first practical admission model should expose:

```text
fixed_request_bytes = sum(recurrent layer state bytes)
token_bytes         = sum(full attention K/V bytes per token)
request_bytes       = fixed_request_bytes + token_bytes * cached_tokens
```

Model weights, CUDA allocator reserve, temporary workspaces, and graph buffers must
be measured separately before choosing pool capacities. `gpu_memory_utilization`
cannot be applied independently to a maximal recurrent pool and a maximal KV pool;
they must share one remaining-memory budget.

For a simple first implementation:

1. profile model/warmup peak memory;
2. reserve an explicit workspace margin;
3. cap recurrent slots to a feasible active-request capacity;
4. allocate remaining memory to Full Attention KV blocks;
5. have scheduler admission require one free recurrent slot and enough initial KV
   blocks;
6. report both resources in diagnostics and benchmarks.

## 14. Modules to modify or add

### Existing modules

| Module | Planned change |
|---|---|
| `nanovllm/config.py` | Extract text config; validate hybrid fields; add state budget controls |
| `nanovllm/engine/sequence.py` | Store stable recurrent-state slot/ownership metadata |
| `nanovllm/engine/block_manager.py` | Keep paged KV behavior; expose transactional/accounting hooks |
| `nanovllm/engine/scheduler.py` | Joint KV + recurrent admission/free/reset; preserve Qwen3 path |
| `nanovllm/engine/model_runner.py` | Model factory, typed state allocation, hybrid batch metadata, eager gating |
| `nanovllm/utils/context.py` | Add recurrent state indices/metadata without model-specific globals |
| `nanovllm/utils/loader.py` | Namespace mapping and strict loaded/missing/duplicate audit |
| `nanovllm/layers/attention.py` | Allow only Full Attention layers to bind paged K/V |

### New modules

The exact names may be adjusted to match the small codebase, but the boundaries
should remain:

```text
nanovllm/models/registry.py
nanovllm/models/qwen3_5.py
nanovllm/layers/gated_deltanet.py
nanovllm/cache/layer_state.py
nanovllm/cache/hybrid_state_manager.py
tests/models/...
tests/cache/...
tests/reference/qwen35_reference.py
benchmarks/qwen35_hybrid/...
```

## 15. Existing behavior that must not regress

- Qwen3 model loading and generation.
- Qwen3 paged KV layout and prefix caching.
- Chunked prefill behavior.
- Decode sampling order and request output order.
- Tensor-parallel code paths, even though Qwen3.5 TP is out of first-phase scope.
- Eager execution and existing CUDA Graph behavior for Qwen3.
- Existing public `LLM` and `SamplingParams` API.

New Qwen3 regression tests are required because the clean baseline has no tests.

## 16. First implementation scope

The first correctness milestone is deliberately narrower than production serving:

- Qwen3.5 dense text-only model, initially single GPU.
- BF16 serving with an FP32 reference and FP32 DeltaNet recurrent state.
- Full Attention through the existing FlashAttention/paged KV path.
- Clear PyTorch Gated DeltaNet reference for prefill and single-token decode.
- Eager execution for Qwen3.5 until state semantics are stable.
- Stable per-request recurrent slots, allocate/free/reset/prepare_batch/reorder.
- Continuous batching with dynamic request completion.
- Prefix cache disabled for Qwen3.5 unless both K/V and recurrent snapshot exist.
- Strict text-weight loading; vision and MTP weights skipped only by explicit policy.

Not in the first milestone: vision, TP/PP, quantization, LoRA, speculative decode,
production preemption, hybrid prefix snapshots, custom Full Attention kernels, or
Qwen3.5 CUDA Graph capture.

## 17. Risk register

| Risk | Impact | Mitigation / required evidence |
|---|---|---|
| Qwen3.5 checkpoint is a multimodal wrapper | Wrong config and weight namespaces | Normalize `text_config`; explicit text allowlist |
| Qwen3.5 RMSNorm differs from Qwen3 | Large error from layer 0 | Dedicated `1 + weight` implementation and unit test |
| Q projection includes attention output gate | Wrong Full Attention output | Match HF shapes and gate placement exactly |
| Partial/interleaved mRoPE differs from Qwen3 | Position-dependent logits drift | Compare RoPE and one Full Attention layer to HF |
| DeltaNet prefill algorithm is complex | Slow reference or numerical drift | Start with explicit recurrence; compare each stage |
| FP32 recurrent state is large | OOM at high `max_num_seqs` | Joint budget and capped slot pool |
| Batch reorder attaches wrong state | Silent cross-request corruption | Stable slot IDs and isolation/reorder tests |
| Prefix cache reuses only K/V | Incorrect generation | Disable hybrid prefix hits in first phase |
| Preemption drops only K/V | Stale recurrent state | Reset both and recompute in first version |
| Chunked prefill resumes from state incorrectly | Diverges on long prompts | Dedicated chunked-vs-one-shot tests |
| Loader silently skips text tensors | Plausible but invalid output | Exact expected/loaded/duplicate sets |
| CUDA Graph captures mutable state incorrectly | Corruption across replay | Qwen3.5 eager-only until graph-specific tests |
| Reference tolerance is too loose | Masks semantic errors | FP32 stage comparison plus justified BF16 tolerance |
| HF optional FLA/causal-conv paths differ | Reference mismatch | Fix one declared HF reference environment and record it |
| No tests exist in baseline | Regressions are easy | Add CPU-safe metadata tests before GPU integration |

## 18. Implementation plan and phase gates

### Phase 0: repository analysis (current phase)

Deliverables:

- clean worktree and feature branch;
- this execution/cache analysis;
- verified Qwen3.5-9B config, layer pattern, state shapes and weight namespaces.

Exit gate: baseline provenance recorded and no model code changed.

Suggested commits:

```text
chore: create clean qwen35 hybrid branch
docs: analyze nano-vllm execution and cache assumptions
```

### Phase 1: model structure and strict weight loading

1. Add a minimal model factory keyed by HF architecture/model type.
2. Normalize wrapper config to `hf_text_config`.
3. Implement Qwen3.5 RMSNorm, partial text RoPE, Full Attention gate, MLP, and
   layer-pattern construction.
4. Implement PyTorch Gated DeltaNet projection/convolution/recurrence reference.
5. Add strict checkpoint namespace mapping and load audit.
6. Compare individual components and full no-cache logits with HF.

Tests:

```text
tests/models/test_qwen35_weight_loading.py
tests/models/test_qwen35_layer_pattern.py
tests/models/test_qwen35_forward.py
```

Exit gate: every required text weight loads once; vision/MTP skips are explicit;
full logits and first divergent layer are reported.

### Phase 2: prefill/decode reference semantics

1. Add `tests/reference/qwen35_reference.py`.
2. Define one-shot and chunked prefill state transitions.
3. Define one-token decode using pre-existing state.
4. Compare batch 1/2/4, variable lengths, early finish and non-contiguous IDs.
5. Record max/mean/relative errors and NaN/Inf at each critical stage.

Exit gate: greedy tokens align with HF over multi-step decode, and chunked prefill
matches one-shot prefill.

### Phase 3: Hybrid Layer State manager

1. Add typed `PagedKVLayerStateSpec` and `RecurrentLayerStateSpec`.
2. Add stable request-to-state-slot allocation.
3. Implement allocate/free/reset/prepare_batch/reorder/get/memory methods.
4. Bind Full Attention layers to paged K/V only.
5. Bind DeltaNet layers to convolution and recurrent pools only.
6. Add state debug counters and ownership assertions.

Tests:

```text
tests/cache/test_hybrid_state_allocate_free.py
tests/cache/test_hybrid_state_reorder.py
tests/cache/test_hybrid_state_isolation.py
tests/cache/test_hybrid_state_memory.py
```

Exit gate: reorder and free/reuse tests prove no request-state leakage.

### Phase 4: Scheduler and continuous batching

1. Introduce joint admission transactions for KV blocks and recurrent slots.
2. Make completion and preemption release/reset both state classes.
3. Disable Qwen3.5 prefix hits explicitly.
4. Validate dynamic joins/exits and differing generation lengths.
5. Add Qwen3 regression tests.

Exit gate: mixed-batch outputs equal per-request isolated runs for batch 1/2/4/8.

### Phase 5: benchmark and profiler harness

Create `benchmarks/qwen35_hybrid/` with correctness, prefill, decode, E2E,
cache-memory, continuous-batching and profiler entry points. Every run writes JSON
and Markdown and records command, commit, model/config, versions, GPU, seed, warmup,
repeat count, mean, p50 and p95.

Exit gate: small-model sweep is reproducible; 9B final commands run on the RTX 5090.

### Phase 6: profiler attribution

Use PyTorch Profiler and Nsight Systems to separate Full Attention, Gated DeltaNet,
Linear/GEMM, state gather/scatter, CPU dispatch and launch overhead for prefill and
decode. Use Nsight Compute only on a confirmed hot kernel.

Exit gate: `docs/qwen35_hybrid/04_profile_analysis.md` points from each conclusion
to checked-in profiler output. No hotspot is presumed before measurement.

### Phase 7: one profiler-selected optimization

Choose one frequent DeltaNet decode fusion only if it has material self time and
intermediate/launch overhead. Implement PyTorch reference, Triton kernel,
correctness sweep, latency sweep, E2E A/B and occupancy/register/memory analysis.

Exit gate: report both local and E2E results, including a negative result if
Amdahl's Law prevents visible E2E improvement.

### Phase 8: hybrid prefix-state design

Write the snapshot design before implementation. Evaluate block-boundary snapshots,
memory cost, hash/version keys, refcounts, eviction, restore semantics and KV-only
versus KV+state hits.

Exit gate: no Hybrid Prefix Cache code until base hybrid serving is stable.

## 19. Immediate next actions

The next coding phase should begin with model/config and strict loading, not cache
optimization:

1. establish a reproducible Transformers reference environment;
2. select a smaller Qwen3.5 dense checkpoint for correctness sweeps while retaining
   Qwen3.5-9B as the final target;
3. add Qwen3 regression smoke tests;
4. implement model registry and text-config normalization;
5. implement Qwen3.5 components incrementally with per-component HF comparisons;
6. commit only after the Phase 1 correctness gate passes.

No Qwen3.5 model code, state manager, scheduler change, benchmark result, or
performance claim is part of this analysis phase.
