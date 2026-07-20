# Qwen3.5 Hybrid Serving Profile Analysis

This file is replaced by measured output from the RTX 5090 profiling run. The
repository does not prefill hotspot percentages or optimization claims before those
traces exist.

## Attribution Model

The profiler records two deliberately separate views:

- Component attribution uses `record_function` ranges around Full Attention,
  DeltaNet convolution/recurrence/output, MLP, metadata preparation, and hybrid-state
  gather/commit. These ranges can contain child kernels and must not be added as
  wall-clock time.
- CUDA kernel self-time uses CUDA device events only. This is the additive view used
  to choose a low-level optimization target.

The generated JSON retains per-case ranges, CUDA kernel categories, top concrete
kernels, CUDA runtime calls, selected allocation/copy/index operators, wall time, and
the raw Chrome trace path.

## PyTorch Profiler Matrix

Run the complete requested matrix on the RTX 5090:

```bash
python benchmarks/qwen35_hybrid/profile_serving.py \
  --model ../models/Qwen3.5-9B \
  --batch-sizes 1,4,8 \
  --prompt-lens 128,512,2048 \
  --decode-steps 32,128 \
  --phases prefill,decode,continuous \
  --warmup 1 \
  --record-shapes \
  --profile-memory
```

This produces 54 Chrome traces under
`benchmarks/qwen35_hybrid/results/traces/`, the machine-readable
`profile_analysis.json`, and overwrites this document with the measured summary.
For a quick environment check, first run one shape:

```bash
python benchmarks/qwen35_hybrid/profile_serving.py \
  --model ../models/Qwen3.5-9B \
  --batch-sizes 1 \
  --prompt-lens 128 \
  --decode-steps 32 \
  --phases prefill,decode,continuous \
  --warmup 1
```

## Nsight Systems

After inspecting the matrix, capture representative prefill, decode, and mixed
continuous-batching timelines. For example:

```bash
python benchmarks/qwen35_hybrid/run_nsight.py \
  --tool nsys \
  --model ../models/Qwen3.5-9B \
  --phase decode \
  --batch-size 4 \
  --prompt-len 512 \
  --decode-steps 128
```

The `.nsys-rep` timeline should be used to check launch gaps, host synchronization,
small-kernel fragmentation, and overlap. Change `--phase` to `prefill` and
`continuous` for the other execution modes.

## Nsight Compute

Run Nsight Compute only for the one or two concrete kernels selected by the PyTorch
profile and Systems timeline:

```bash
python benchmarks/qwen35_hybrid/run_nsight.py \
  --tool ncu \
  --model ../models/Qwen3.5-9B \
  --phase decode \
  --batch-size 4 \
  --prompt-len 512 \
  --decode-steps 128 \
  --kernel-name 'EXACT_OR_REGEX_FROM_PROFILE' \
  --launch-skip 0 \
  --launch-count 1
```

Use `--dry-run` to inspect either command without launching Nsight. Nsight Compute
should answer a kernel-specific question such as memory throughput, occupancy,
instruction mix, or launch configuration; it is not run across the full matrix.

## Decision Rule

No Triton work starts in this phase. The next target is selected only after the data
answers all of the following:

1. Full Attention component attribution by phase and shape.
2. DeltaNet component attribution and recurrence call granularity.
3. Whether Linear/GEMM remains the largest CUDA leaf-kernel category.
4. Whether recurrent updates create many short kernels and launch gaps.
5. Whether hybrid-state gather/commit is material in CUDA or CPU time.
6. How attribution changes from batch 1 to 4 to 8.
7. How Full Attention changes from prompt 128 to 512 to 2048.
8. Which single measured kernel is the best next optimization candidate.

Profiler overhead means these traces explain bottleneck shape; Phase 4's repeated E2E
benchmark remains the source for uninstrumented latency and throughput.
