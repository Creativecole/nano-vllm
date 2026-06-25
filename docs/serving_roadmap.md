# Serving Roadmap

This fork currently focuses on offline generation and inference-engine profiling. A minimal online
serving layer is useful for portfolio completeness, but it should not distract from the main kernel and
runtime analysis work.

## Current Offline Path

The current public path is:

```text
LLM.generate(prompts, sampling_params)
```

Internally, the engine still has the right serving pieces:

- `LLMEngine.add_request`
- `LLMEngine.step`
- `Scheduler.waiting`
- `Scheduler.running`
- paged KV-cache block allocation
- prefix-cache reuse
- decode CUDA Graph replay

This means an HTTP layer can be built on top without rewriting the model path.

## Minimal Online Serving Prototype

Target endpoints:

| Endpoint | Scope |
|---|---|
| `POST /v1/completions` | First target; prompt string plus max tokens and temperature |
| `POST /v1/chat/completions` | Optional wrapper that formats chat messages into a prompt |

Initial behavior:

- Single model loaded at process start.
- Request queue feeding `LLMEngine.add_request`.
- Background loop repeatedly calling `LLMEngine.step`.
- Non-streaming response first.
- Basic latency logging: queue time, TTFT, total latency, generated tokens.

## Continuous Batching

The scheduler already separates waiting and running sequences. An online server needs to connect that
to request arrival:

1. Accept HTTP request and tokenize prompt.
2. Add request to waiting queue.
3. Run scheduler loop continuously.
4. Resolve each HTTP future when its sequence finishes.

The hard part is not the endpoint itself. The hard part is clean lifecycle handling: cancellation,
timeouts, backpressure, max queue size, and graceful shutdown.

## Streaming

Streaming requires token-level visibility:

- Track new tokens per sequence after each decode step.
- Push deltas to the client through SSE or chunked responses.
- Preserve final usage statistics.

This needs a small extension to the current output collection path because `generate()` currently
returns only finished completions.

## Metrics For Serving

An online benchmark should report:

| Metric | Meaning |
|---|---|
| TTFT | Request arrival to first streamed/generated token |
| TPOT / ITL | Per-output-token latency after first token |
| p50/p95 latency | Distribution across concurrent requests |
| Queue time | Time waiting before first scheduled prefill |
| Tokens/s | Aggregate serving throughput |
| KV utilization | Max KV-cache pressure under concurrent load |
| Prefix hit rate | Effectiveness of shared system prompts |

## Suggested Implementation Steps

1. Add a small FastAPI server around one `LLM` instance.
2. Implement `POST /v1/completions` without streaming.
3. Add structured request/response latency logging.
4. Add a benchmark script that sends concurrent requests.
5. Add streaming only after non-streaming correctness and cancellation behavior are stable.

The serving layer should remain optional. The inference-engine benchmark/profiling path is still the
core of this project.
