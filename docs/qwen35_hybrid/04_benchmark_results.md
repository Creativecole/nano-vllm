# Qwen3.5 Hybrid E2E Benchmark

This file is generated from measured RTX 5090 runs. No placeholder performance
number is presented as a result.

```bash
python benchmarks/qwen35_hybrid/bench_e2e.py \
  --model ../models/Qwen3.5-9B \
  --backends hf,nanovllm,vllm \
  --batch-sizes 1,2,4,8 \
  --prompt-lens 128,512,2048 \
  --output-lens 32,128 \
  --warmup 1 \
  --repeat 5
```

Each backend runs in an isolated process. The generated JSON stores raw repeats,
mean/p50/p95 aggregates, failed configurations, peak memory, active full-attention
KV bytes, and DeltaNet recurrent-state bytes.
