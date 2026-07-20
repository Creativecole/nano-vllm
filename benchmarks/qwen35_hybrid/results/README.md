# Qwen3.5 Hybrid Benchmark Artifacts

The Qwen3.5 validation and profiling scripts write machine-readable artifacts into
this directory. No performance or hotspot number is checked in until it has been
produced by an actual model run.

Expected Phase 5 outputs:

- `profile_analysis.json`: full PyTorch Profiler matrix and derived attribution.
- `traces/*.json`: raw Chrome traces for prefill, decode, and continuous batching.
- `nsight/*.nsys-rep`: selected Nsight Systems timelines.
- `nsight/*.ncu-rep`: one or two profiler-selected Nsight Compute kernel reports.

The generated human-readable summary is
`docs/qwen35_hybrid/05_profile_analysis.md`. Large raw traces should be archived for
analysis and do not need to be committed to Git.
