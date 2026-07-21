# Qwen3.5 DeltaNet Kernel Fragmentation

## Root Cause From Source

The current nano-vLLM Gated DeltaNet implementation is a correctness-first PyTorch
reference. `gated_delta_rule_reference` contains an explicit
`for token_idx in range(seq_len)` loop. For every token it separately launches work
for decay, state retrieval, two reductions, delta construction, recurrent-state
update, and output retrieval. Packed prefill additionally loops over requests before
entering the token recurrence.

For Qwen3.5-9B, the configured 24 DeltaNet layers and a 2048-token prompt produce
49,152 token-layer recurrence iterations. Approximately five elementwise and two
reduction operations per iteration predict about 245,760 elementwise and 98,304
reduction launches. This closely matches the observed whole-model profile:

| Event | Calls / time |
|---|---:|
| `cudaLaunchKernel` | 578,739 calls / 2,424 ms CPU time |
| `elementwise_kernel` | 255,009 calls / 476 ms CUDA time |
| `reduce_kernel` | 99,792 calls / 311 ms CUDA time |

The count correspondence makes the sequential recurrence the leading explanation for
the prefill launch explosion. Causal convolution is one operation per request/layer,
so it cannot by itself explain growth proportional to `layers * sequence_length`.

## Layer-Level Experiment

The isolated profiler compares the real packed serving-prefill token mixers. Full
Attention uses paged KV storage plus FlashAttention varlen; DeltaNet uses the stateful
packed path. Decoder MLP and outer RMSNorm are excluded so their GEMMs do not obscure
the mixer comparison.

The commands below describe the original sequential-only run. The profiler now also
supports the chunked reference backend; use the Phase 6 commands in
`06_chunked_recurrence.md` for a side-by-side run without replacing this baseline.

Run the first configured layer of each type at batch 1:

```bash
python benchmarks/qwen35_hybrid/profile_layers.py \
  --model ../models/Qwen3.5-9B \
  --batch-size 1 \
  --prompt-len 128 \
  --prompt-len 512 \
  --prompt-len 2048 \
  --warmup 1 \
  --repeat 1 \
  --record-shapes \
  --profile-memory
```

To inspect explicit layer IDs, repeat `--layer-id`:

```bash
python benchmarks/qwen35_hybrid/profile_layers.py \
  --model ../models/Qwen3.5-9B \
  --layer-id 0 \
  --layer-id 3 \
  --batch-size 1 \
  --prompt-len 128 \
  --prompt-len 512 \
  --prompt-len 2048
```

The original run atomically checkpointed:

- `benchmarks/qwen35_hybrid/results/layer_profile.json`
- `benchmarks/qwen35_hybrid/results/traces/layers/*.json`
- this Markdown file with the measured comparison table

The current comparison schema intentionally writes to `deltanet_backend_profile.json`
instead, so the measured sequential baseline remains available.

## Questions To Resolve From Layer Data

1. Compare `kernel_count_per_forward` for Full Attention and DeltaNet.
2. Check whether DeltaNet kernel count grows linearly from 128 to 512 to 2048.
3. Compare `qwen35_deltanet_conv`, `qwen35_deltanet_recurrence`, and
   `qwen35_deltanet_output` range attribution.
4. Verify whether recurrence accounts for the elementwise/reduction kernels while
   convolution remains approximately constant in launch count.
5. Use `top_cuda_kernels` and `top_operators` to identify fusion boundaries.

Likely fusion candidates are Q/K normalization and the recurrent
decay/retrieval/delta/state/output sequence. Projection GEMMs and causal convolution
should remain library operations unless the measured layer profile says otherwise.

Hugging Face and production vLLM fast paths can dispatch chunked/fused Gated DeltaNet
and optimized causal-convolution or recurrent kernels. They avoid a Python-controlled
chain of elementwise and reduction launches for every token. This phase only measures
that gap; it does not modify model logic or add a Triton kernel.
