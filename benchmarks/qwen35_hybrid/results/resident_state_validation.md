# Qwen3.5 Resident DeltaNet State Validation

This artifact separates the online serving result from the fixed-shape PyTorch
Profiler result. Both compare the same resident-state implementation against the
materialized `index_select` / `index_copy_` fallback.

## Online A/B

Environment:

- Model: Qwen3.5-9B, BF16, eager mode
- GPU: NVIDIA RTX 5090
- Workload: seeded mixed open-loop workload
- Request rate: 1 request/s
- Arrival window: 60 seconds
- Scheduler: `prefill_first`
- DeltaNet backend: `chunked`, chunk size 64

| State path | Completed | Rejected | TTFT p50 ms | TTFT p95 ms | ITL p95 ms | Output tok/s | Gather calls | Commit calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gather/commit | 53 | 0 | 858.373 | 3693.169 | 44.291 | 90.608 | 1096 | 1096 |
| Resident | 53 | 0 | 818.945 | 3619.757 | 44.020 | 90.655 | 0 | 0 |

The two paths complete the same number of requests without rejection. Throughput is
effectively tied. Resident state lowers TTFT p50 by 4.6% in this single seeded run, but
the latency delta should not be treated as a general claim without repeated seeds.

The runtime counters estimate 569.25 GiB of logical gather/commit traffic avoided and
2.37 GiB copied during request-completion compaction. These are tensor-size estimates,
not measured DRAM traffic.

## Fixed-Shape Profiler A/B

The profiler case uses batch 4, prompt length 512, 64 decode steps, and the continuous
phase.

| Metric | Gather/commit | Resident | Change |
|---|---:|---:|---:|
| `aten::index_select` calls | 3185 | 65 | -98.0% |
| `aten::index_select` self CUDA | 20.300 ms | 0.156 ms | -99.2% |
| `aten::index_copy_` calls | 3120 | 0 | eliminated |
| `aten::index_copy_` self CUDA | 16.832 ms | 0 ms | eliminated |
| State gather attributed CUDA | 77.448 ms | 0 ms | eliminated |
| State commit attributed CUDA | 17.655 ms | 0 ms | eliminated |
| `cudaLaunchKernel` calls | 210387 | 204051 | -3.0% |
| CUDA runtime self CPU | 1097.634 ms | 1068.176 ms | -2.7% |
| Kernel self CUDA total | 16144.378 ms | 15843.397 ms | -1.9% |
| Profile wall time | 22.863 s | 22.389 s | -2.1% |

`aten::copy_` remains essentially unchanged because most copies belong to model
execution rather than DeltaNet state materialization. The result validates the resident
state mechanism and rules out state gather/commit as the primary serving bottleneck for
this workload.

## Interpretation

Resident state is kept as the default path because it removes avoidable state
materialization without increasing peak memory or reducing throughput. The fallback is
retained for correctness checks. Further performance work should target measured model
execution and scheduling bottlenecks rather than assuming state movement dominates.
