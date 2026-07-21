# Qwen3.5-9B Hybrid Serving on RTX 5090

A text-only Qwen3.5 hybrid inference backend built inside nano-vLLM. It combines
Paged KV Cache for Full Attention layers with request-scoped convolution and recurrent
state for Gated DeltaNet layers, then optimizes prefill from token-level recurrence to
chunked and truly batched execution.

The measured implementation runs Qwen3.5-9B in BF16 on one NVIDIA RTX 5090. It does
not use quantization, a custom Triton kernel, or CUDA Graphs for these results.

## Project Branches

The two projects intentionally remain on separate branches:

| Branch | Focus |
|---|---|
| [`qwen35-hybrid`](https://github.com/Creativecole/nano-vllm/tree/qwen35-hybrid) | **This project:** Qwen3.5-9B hybrid state, DeltaNet prefill, continuous batching, profiling |
| [`main`](https://github.com/Creativecole/nano-vllm/tree/main) | Qwen3-4B decode-only Triton PagedAttention backend |

## Results

Measured with Qwen3.5-9B, BF16, eager mode, one RTX 5090, warmup 2, repeat 3, and
128 generated tokens. Hugging Face and nano-vLLM run in isolated processes. The HF
runtime was verified to use `causal_conv1d_fn` and FLA
`chunk_gated_delta_rule` during prefill.

### End-to-End Serving

| Batch | Prompt | HF TTFT | nano TTFT | HF avg ITL | nano avg ITL | HF decode tok/s | nano decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 176.7 ms | **169.9 ms** | 47.78 ms | **41.41 ms** | 20.93 | **24.15** |
| 1 | 512 | 214.5 ms | **200.5 ms** | 47.91 ms | **41.04 ms** | 20.87 | **24.37** |
| 1 | 2048 | 360.6 ms | **331.4 ms** | 47.99 ms | **41.24 ms** | 20.84 | **24.25** |
| 4 | 128 | 181.5 ms | **175.3 ms** | 48.64 ms | **41.98 ms** | 82.24 | **95.28** |
| 4 | 512 | 240.8 ms | **235.0 ms** | 48.84 ms | **42.14 ms** | 81.90 | **94.93** |
| 4 | 2048 | 999.5 ms | **952.2 ms** | 48.85 ms | **42.14 ms** | 81.88 | **94.92** |

On this matrix, nano-vLLM is 2.4% to 8.1% lower in TTFT and about 16% higher in
aggregate decode throughput than the measured HF eager baseline. These are runtime
results for this exact model, GPU, and workload, not general claims across hardware.
ITL is average inter-token latency, while decode tok/s is aggregate throughput across
the batch; the batch-4 throughput is not per-request generation speed.

### Batched Prefill Optimization

The first chunked implementation still split packed requests and called recurrence once
per request per DeltaNet layer. The equal-length fast path now preserves the batch
dimension and performs one batched recurrence call per layer.

| Batch | Prompt | Before batched path | Batched path | Improvement |
|---:|---:|---:|---:|---:|
| 4 | 128 | 607 ms | **175.3 ms** | **3.46x** |
| 4 | 512 | 739 ms | **235.0 ms** | **3.14x** |
| 4 | 2048 | 1429 ms | **952.2 ms** | **1.50x** |

Batch-1 TTFT stays effectively unchanged: 170.0 to 169.9 ms at prompt 128,
201.0 to 200.5 ms at prompt 512, and 330.0 to 331.4 ms at prompt 2048. The earlier
baseline used repeat 2; the final table uses repeat 3, both after two warmups.

### Why Chunked Recurrence

The original correctness-first recurrence loop launched roughly ten CUDA kernels per
input token. A PyTorch chunked Gated Delta Rule keeps the same equations and FP32 state
accumulation but evaluates within-chunk dependencies with batched matrix operations.

Single DeltaNet layer, batch 1:

| Prompt | Sequential CUDA | Chunked CUDA | Sequential kernels | Chunked kernels | CUDA speedup |
|---:|---:|---:|---:|---:|---:|
| 128 | 57.23 ms | 22.92 ms | 1,338 | 502 | 2.50x |
| 512 | 222.24 ms | 29.50 ms | 5,181 | 604 | 7.53x |
| 2048 | 889.79 ms | 75.62 ms | 20,553 | 1,012 | 11.77x |

At prompt 2048, chunking removes about 95% of the layer-level kernel launches. This is
an execution-model optimization implemented with PyTorch operations, not a fused Triton
kernel claim.

## Architecture

Qwen3.5-9B text configuration:

- 32 decoder layers with pattern `[DeltaNet, DeltaNet, DeltaNet, Full Attention] x 8`
- 24 Gated DeltaNet layers and 8 Full Attention layers
- Full Attention: 16 query heads, 4 KV heads, head dimension 256
- DeltaNet: 16 key heads, 32 value heads, key/value head dimension 128
- Causal convolution width 4
- Hidden size 4096, MLP intermediate size 12288
- BF16 model tensors with FP32 recurrent-state accumulation

### Hybrid Cache

Full Attention layers use nano-vLLM's paged KV blocks and block tables. DeltaNet layers
use a separate fixed-slot state pool indexed by request ID:

```text
PagedKVState  = K/V blocks for 8 Full Attention layers
DeltaNetState = conv_state + recurrent_state for 24 DeltaNet layers
```

The scheduler accounts token capacity with Full Attention KV blocks. DeltaNet state is
allocated independently per active request, gathered in active batch order, committed
after each model step, reordered during batch compaction, and released when the request
finishes.

For Qwen3.5-9B, active DeltaNet state is about 49.5 MiB per request. nano-vLLM also
preallocates a large KV cache pool, so its reported peak GPU allocation is higher than
HF eager and should not be interpreted as prompt-specific live KV usage.

## Implementation

### Model Integration

- model registry removes the original hard-coded Qwen3 model construction
- outer `Qwen3_5Config` and nested text config are handled separately
- layer construction follows the checkpoint's `layer_types`
- Full Attention includes GQA, attention gate, partial RoPE, and paged KV execution
- Gated DeltaNet includes projections, causal convolution, decay/gate, recurrent state,
  gated normalization, and output projection
- strict weight loading reports loaded, missing, duplicate, unexpected, and intentionally
  skipped non-text weights

### Prefill and Decode

```text
Prefill
  packed requests
    -> Full Attention: paged/varlen attention
    -> DeltaNet: chunked recurrence
    -> equal-length batch: one [B, T, ...] recurrence call per layer
    -> variable length: correctness-first per-sequence fallback

Decode
  one token per active request
    -> Full Attention: paged KV read/write
    -> DeltaNet: sequential recurrent-state update
```

The `sequential` DeltaNet backend remains available as the reference implementation.
The optimized path is selected explicitly with `deltanet_backend="chunked"`.

## Quick Start

### Installation

```bash
git clone --branch qwen35-hybrid \
  https://github.com/Creativecole/nano-vllm.git
cd nano-vllm
pip install -e .
```

The measured environment used Python 3.11, PyTorch 2.12.0+cu130, CUDA 13.0,
Transformers 5.10.2, and FlashAttention on an RTX 5090.

Download the model:

```bash
hf download Qwen/Qwen3.5-9B --local-dir ../models/Qwen3.5-9B
```

Run generation:

```python
from nanovllm import LLM, SamplingParams

llm = LLM(
    "../models/Qwen3.5-9B",
    enforce_eager=True,
    max_num_seqs=4,
    hybrid_state_capacity=4,
    deltanet_backend="chunked",
    deltanet_chunk_size=64,
)
outputs = llm.generate(
    [[1, 2, 3, 4]],
    SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True),
    use_tqdm=False,
)
llm.exit()
```

## Project Layout

```text
nanovllm/
  models/qwen3_5.py          # Full Attention + Gated DeltaNet model
  engine/layer_state.py      # Paged KV and request-scoped DeltaNet state
  engine/model_runner.py     # hybrid cache allocation and execution
  models/registry.py         # model dispatch

benchmarks/qwen35_hybrid/
  validate_checkpoint.py
  validate_correctness.py
  diagnose_batched_prefill.py
  bench_e2e.py
  profile_layers.py
  profile_serving.py
  run_nsight.py

docs/qwen35_hybrid/          # design, correctness, profiler, and optimization notes
tests/                       # model, state lifecycle, scheduler, and benchmark tests
```

## License

MIT
