# Qwen3.5 Chunked Recurrence: Final Validation

## Problem

Qwen3.5-9B contains 24 Gated DeltaNet layers and 8 Full Attention layers. The original
correctness-first DeltaNet prefill path evaluated the recurrence with a Python loop over
every token. Each iteration launched several PyTorch elementwise and reduction kernels,
so launch count scaled approximately with `batch * prompt_length`.

The chunked backend keeps the same Gated Delta Rule, BF16 model I/O, and FP32 recurrent
state, but evaluates within-chunk dependencies with matrix operations and updates the
recurrent state once per chunk. Single-token decode still uses the sequential update.
No Triton kernel or model-architecture change is part of this optimization.

## Environment And Method

- Model: Qwen3.5-9B text-only
- GPU: NVIDIA GeForce RTX 5090
- Dtype: BF16; recurrent state accumulation: FP32
- PyTorch / CUDA: 2.12.0+cu130 / 13.0
- Transformers: 5.10.2
- DeltaNet layer: layer 0
- Chunk size: 64
- Layer profile: one warmup and one measured forward with PyTorch Profiler
- E2E: one warmup and one measured run, output length 32

The E2E data is a single measured repeat. It confirms the execution-model effect, but a
repeat-5 run should be used before quoting low-noise p50/p95 latency.

## Prompt-Length Scaling

Batch size is 1.

| Prompt | Sequential CUDA | Chunked CUDA | CUDA speedup | Sequential kernels | Chunked kernels | Kernel reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 57.23 ms | 22.92 ms | 2.50x | 1,338 | 502 | 62.48% |
| 512 | 222.24 ms | 29.50 ms | 7.53x | 5,181 | 604 | 88.34% |
| 2048 | 889.79 ms | 75.62 ms | 11.77x | 20,553 | 1,012 | 95.08% |

Sequential execution stays near 10 kernels per input token. Chunked execution amortizes
its fixed chunk transform as the prompt grows, reducing the 2048-token layer profile to
about 0.49 kernels per input token.

## Batch Scaling At Prompt 2048

| Batch | Sequential CUDA | Chunked CUDA | CUDA speedup | Sequential kernels | Chunked kernels | Kernel reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 889.79 ms | 75.62 ms | 11.77x | 20,553 | 1,012 | 95.08% |
| 4 | 3,671.90 ms | 215.76 ms | 17.02x | 82,206 | 4,042 | 95.08% |
| 8 | 7,121.26 ms | 408.42 ms | 17.44x | 164,410 | 8,082 | 95.08% |

Kernel count still scales with packed request count because the serving prefill path
processes each request's independent DeltaNet state separately. The important change is
that each request now scales by chunks rather than by token-level recurrence launches.

## End-To-End Effect

Batch size is 1 and output length is 32.

| Prompt | Sequential TTFT | Chunked TTFT | TTFT speedup | Sequential decode tok/s | Chunked decode tok/s | Decode ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 394.08 ms | 178.78 ms | 2.20x | 23.96 | 24.16 | 1.008x |
| 512 | 1,431.22 ms | 204.51 ms | 7.00x | 23.98 | 24.05 | 1.003x |
| 2048 | 5,532.45 ms | 334.45 ms | 16.54x | 24.22 | 23.97 | 0.990x |

The improvement appears in TTFT because chunked recurrence changes prefill. Decode
throughput remains effectively unchanged, which is expected because one-token decode
continues to use the sequential recurrent state update.

## Memory Tradeoff

| Workload | Sequential peak delta | Chunked peak delta | Extra temporary memory |
|---|---:|---:|---:|
| Layer, batch 1, prompt 128 | 26.16 MiB | 40.65 MiB | 14.49 MiB |
| Layer, batch 1, prompt 512 | 88.30 MiB | 133.85 MiB | 45.55 MiB |
| Layer, batch 1, prompt 2048 | 340.77 MiB | 518.24 MiB | 177.47 MiB |
| Layer, batch 4, prompt 2048 | 388.77 MiB | 566.73 MiB | 177.97 MiB |
| Layer, batch 8, prompt 2048 | 452.77 MiB | 630.73 MiB | 177.97 MiB |

At E2E prompt 2048, chunked peak allocated memory is 28.196 GiB versus 28.022 GiB for
sequential, an increase of about 0.174 GiB. The execution-model speedup therefore trades
additional chunk intermediates for substantially fewer launches and lower TTFT.

## Nsight Systems Validation

The supplied artifacts contain PyTorch Profiler data but no `.nsys-rep` timeline yet.
Generate the two captures with identical model and workload settings:

```bash
python benchmarks/qwen35_hybrid/run_nsight.py \
  --tool nsys \
  --model ../models/Qwen3.5-9B \
  --phase prefill \
  --batch-size 1 \
  --prompt-len 2048 \
  --decode-steps 1 \
  --warmup 1 \
  --deltanet-backend sequential \
  --deltanet-chunk-size 64 \
  --output benchmarks/qwen35_hybrid/results/nsight/nsys_deltanet_seq_b1_p2048

python benchmarks/qwen35_hybrid/run_nsight.py \
  --tool nsys \
  --model ../models/Qwen3.5-9B \
  --phase prefill \
  --batch-size 1 \
  --prompt-len 2048 \
  --decode-steps 1 \
  --warmup 1 \
  --deltanet-backend chunked \
  --deltanet-chunk-size 64 \
  --output benchmarks/qwen35_hybrid/results/nsight/nsys_deltanet_chunked_b1_p2048
```

Expected outputs:

- `nsys_deltanet_seq_b1_p2048.nsys-rep`
- `nsys_deltanet_chunked_b1_p2048.nsys-rep`

Inspect the `qwen35_deltanet_recurrence_sequential` and
`qwen35_deltanet_recurrence_chunked` NVTX ranges. The validation criterion is visible
removal of the token-level chain of small kernels and a more compact chunk-level
timeline. This observation must be filled in only after opening the generated reports.

Use `--dry-run` on either command to print the exact `nsys profile` invocation without
executing it. The same backend and chunk-size arguments are also forwarded for `ncu`.

## Conclusion

The measured bottleneck was an inefficient execution model, not evidence that a new GPU
kernel was immediately required. Replacing token-level Python recurrence with the
official-style PyTorch chunked formulation reduced layer kernel count by up to 95% and
improved layer CUDA time by up to 17.44x in the tested matrix. The corresponding E2E
effect is lower TTFT, while decode throughput remains stable.

This supports a profile-first sequence: fix launch fragmentation with a validated
execution-model change, quantify its memory cost, and only then decide whether a fused
CUDA or Triton kernel is justified.
