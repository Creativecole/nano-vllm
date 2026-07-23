# Qwen3.5 Hybrid Serving Saturation Analysis

## Methodology

- Model: Qwen3.5-9B, BF16, eager mode
- GPU: NVIDIA RTX 5090
- Runtime: nano-vLLM chunked DeltaNet prefill, chunk size 64
- Workload: seeded Poisson open-loop mixed workload
- Mix: 80% chat, 20% long context
- Arrival window: 60 seconds
- Configured request rates: 0.5, 1, 2, 4, and 8 requests/s
- Limits: 8 inflight requests, 32 queued requests
- Seed: 17

All offered requests, failures and rejections remain visible. Throughput uses the full
wall time, including draining admitted requests after the arrival window.

## Results

| Request rate | Completed | Rejected | Rejection rate | Input tok/s | Output tok/s | TTFT p50 ms | TTFT p95 ms | ITL p95 ms | Peak pending queue | Drain time s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | 30 | 0 | 0.0% | 981.84 | 52.69 | 160.40 | 1148.23 | 31.70 | 0 | 1.95 |
| 1 | 53 | 0 | 0.0% | 1474.71 | 92.99 | 221.15 | 1744.73 | 32.57 | 3 | 3.32 |
| 2 | 115 | 0 | 0.0% | 2109.67 | 156.51 | 7719.86 | 20679.66 | 36.38 | 32 | 23.42 |
| 4 | 123 | 111 | 47.4% | 1954.68 | 168.78 | 19502.24 | 21942.90 | 33.66 | 32 | 23.80 |
| 8 | 120 | 337 | 73.7% | 2156.26 | 160.01 | 21562.35 | 24120.86 | 32.78 | 32 | 24.40 |

## Interpretation

**Stable region, 0.5-1 request/s.** All requests complete without rejection, the peak
pending queue stays below four, and TTFT p95 remains below 1.8 seconds.

**Saturation knee, 2 requests/s.** Output throughput rises to 156.5 tokens/s, but the
pending queue reaches its limit and TTFT p95 increases to 20.7 seconds. The 23.4-second
drain time confirms that arrivals outpace service during part of the run.

**Overloaded region, 4-8 requests/s.** Output throughput plateaus around 160-169
tokens/s while rejection reaches 47-74%. Additional offered load increases queueing and
rejection rather than useful throughput.

ITL p95 stays within 31.7-36.4 ms across the sweep. The large TTFT increase therefore
comes mainly from admission and prefill waiting, not a comparable degradation in
steady-state decode token spacing.

This sweep was captured before the resident DeltaNet state cleanup. A later controlled
rate-1 A/B measured 90.61 versus 90.66 output tokens/s for gather/commit and resident
state, respectively. The saturation sweep is therefore retained as workload-shape
evidence, while resident-state effects are reported separately in
[`resident_state_validation.md`](resident_state_validation.md).
