#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks/qwen35_hybrid/results/nsight"
PROFILE_SCRIPT = REPO_ROOT / "benchmarks/qwen35_hybrid/profile_serving.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run one Qwen3.5 hybrid serving case under Nsight Systems or Compute."
    )
    parser.add_argument("--tool", required=True, choices=("nsys", "ncu"))
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--phase", required=True, choices=("prefill", "decode", "continuous")
    )
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--prompt-len", type=int, required=True)
    parser.add_argument("--decode-steps", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--output",
        help="Output prefix without .nsys-rep/.ncu-rep; a descriptive default is used.",
    )
    parser.add_argument(
        "--kernel-name",
        help="Nsight Compute regex for one profiler-selected kernel.",
    )
    parser.add_argument("--launch-skip", type=int, default=0)
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def target_command(args) -> list[str]:
    return [
        sys.executable,
        str(PROFILE_SCRIPT),
        "--model",
        args.model,
        "--target-only",
        "--target-phase",
        args.phase,
        "--target-batch",
        str(args.batch_size),
        "--target-prompt",
        str(args.prompt_len),
        "--target-decode",
        str(args.decode_steps),
        "--warmup",
        str(args.warmup),
    ]


def output_prefix(args) -> Path:
    if args.output:
        return Path(args.output)
    label = (
        f"qwen35_{args.phase}_b{args.batch_size}_p{args.prompt_len}"
        f"_d{args.decode_steps}_{args.tool}"
    )
    return DEFAULT_RESULTS_DIR / label


def build_command(args) -> list[str]:
    output = output_prefix(args)
    target = target_command(args)
    if args.tool == "nsys":
        return [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "--trace=cuda,nvtx,osrt",
            "--sample=none",
            "--output",
            str(output),
            *target,
        ]
    if not args.kernel_name:
        raise ValueError(
            "--kernel-name is required for ncu; select one of the top profiler kernels"
        )
    return [
        "ncu",
        "--force-overwrite",
        "--target-processes",
        "all",
        "--set",
        "full",
        "--kernel-name-base",
        "demangled",
        "--kernel-name",
        f"regex:{args.kernel_name}",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--export",
        str(output),
        *target,
    ]


def main():
    args = parse_args()
    command = build_command(args)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    if shutil.which(args.tool) is None:
        raise RuntimeError(
            f"{args.tool!r} was not found. Install NVIDIA Nsight tooling or use --dry-run."
        )
    output_prefix(args).parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
