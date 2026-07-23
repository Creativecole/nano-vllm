# Qwen3.5 Hybrid Benchmark Artifacts

The Qwen3.5 validation and profiling scripts write machine-readable artifacts into
this directory. No performance or hotspot number is checked in until it has been
produced by an actual model run.

Expected Phase 5 outputs:

- `profile_analysis.json`: full PyTorch Profiler matrix and derived attribution.
- `layer_profile.json`: isolated Full Attention versus DeltaNet serving-prefill layers.
- `traces/*.json`: raw Chrome traces for prefill, decode, and continuous batching.
- `nsight/*.nsys-rep`: selected Nsight Systems timelines.
- `nsight/*.ncu-rep`: one or two profiler-selected Nsight Compute kernel reports.

Expected chunked-reference validation outputs:

- `chunked_correctness.json`: HF logits and greedy-token comparison.
- `deltanet_backend_profile.json`: sequential/chunked layer CUDA time, kernel count,
  and temporary-memory comparison.
- `chunked_e2e.json`: isolated-process TTFT, decode, and peak-memory comparison.

Resident-state validation:

- `resident_state_validation.md`: low-load online A/B and fixed-shape profiler summary.
- Raw serving and profiler JSON files are intentionally kept outside Git unless needed
  to reproduce a published table; large profiler traces remain local artifacts.

The generated human-readable summary is
`docs/qwen35_hybrid/05_profile_analysis.md`. Large raw traces should be archived for
analysis and do not need to be committed to Git.

`profile_analysis.json` and the Markdown summary are atomically updated after every
completed case. Re-running an identical command resumes the matrix and skips completed
case keys; a different matrix must use another output path or `--no-resume`.
