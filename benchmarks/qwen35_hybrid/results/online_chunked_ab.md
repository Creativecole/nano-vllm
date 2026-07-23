# Qwen3.5 Online Sequential vs Chunked A/B

## Methodology

- Model: Qwen3.5-9B, BF16, eager mode
- GPU: NVIDIA RTX 5090
- Backends: `nanovllm_sequential` and `nanovllm_chunked`
- Scheduler: `prefill_first`
- State path: resident DeltaNet state
- Workload: seeded Poisson mixed workload, 80% chat and 20% long context
- Configured request rate: 0.5 requests/s
- Arrival window: 120 seconds
- Seeds: 17, 29, and 43
- Limits: 8 inflight requests, 32 queued requests

Each backend runs in an isolated process. Backends with the same seed receive the same
generated request trace. All completed, failed and rejected requests remain visible.

## Results

| Seed | Backend | Offered | Completed | Rejected | TTFT p50 ms | TTFT p95 ms | ITL p95 ms | Output tok/s | Peak queue | Drain s | Peak memory GiB |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | Sequential | 53 | 53 | 0 | 114848.39 | 166658.68 | 45.22 | 19.82 | 32 | 177.01 | 27.47 |
| 17 | Chunked | 53 | 53 | 0 | 222.42 | 1172.98 | 43.10 | 47.71 | 1 | 3.42 | 28.16 |
| 29 | Sequential | 65 | 57 | 8 | 112919.06 | 162867.63 | 43.49 | 22.46 | 32 | 162.09 | 27.44 |
| 29 | Chunked | 65 | 65 | 0 | 277.40 | 1249.83 | 43.07 | 58.59 | 1 | 4.53 | 28.12 |
| 43 | Sequential | 66 | 66 | 0 | 53389.72 | 74609.67 | 43.85 | 39.06 | 29 | 83.16 | 27.40 |
| 43 | Chunked | 66 | 66 | 0 | 219.21 | 815.55 | 43.61 | 65.26 | 0 | 1.61 | 28.04 |

Per-seed changes:

| Seed | TTFT p50 reduction | TTFT p95 reduction | Output throughput ratio | ITL p95 reduction |
|---:|---:|---:|---:|---:|
| 17 | 99.81% | 99.30% | 2.41x | 4.68% |
| 29 | 99.75% | 99.23% | 2.61x | 0.97% |
| 43 | 99.59% | 98.91% | 1.67x | 0.56% |

## Interpretation

The sequential backend is a correctness-first token-level recurrence. The mixed
long-context workload pushes it into saturation even at 0.5 configured requests/s,
which is why its TTFT is measured in tens to hundreds of seconds. Chunked prefill
changes the execution model and brings the same offered workload back into the stable
region.

ITL changes little because decode remains a one-token sequential recurrent update in
both backends. The online improvement is therefore attributed to prefill capacity and
queue reduction, consistent with the isolated layer profiler and offline TTFT results.

One sequential run rejects eight requests, so aggregate means across all three seeds
would compare different admitted sets. The report keeps per-seed results visible rather
than presenting one misleading aggregate percentage.
