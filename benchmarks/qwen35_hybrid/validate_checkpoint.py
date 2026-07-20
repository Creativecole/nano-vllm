#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    environment_metadata,
    load_model_facts,
    markdown_table,
    safe_error,
    write_json,
    write_text,
)
from nanovllm.config import Config  # noqa: E402
from nanovllm.models.registry import get_model_class  # noqa: E402
from nanovllm.utils.loader import WeightLoadingError, load_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit real Qwen3.5 text checkpoint coverage without running inference."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "checkpoint_coverage.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(DEFAULT_RESULTS_DIR / "checkpoint_coverage.md"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "environment": environment_metadata(args.model),
        "model_facts": load_model_facts(args.model),
    }
    config = Config(args.model, enforce_eager=True)
    model_class = get_model_class(config.hf_config, config.hf_text_config)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(config.dtype)
        print(f"[coverage] constructing {model_class.__name__} on CPU", flush=True)
        model = model_class(config.hf_text_config)
        print("[coverage] scanning and loading safetensor shards", flush=True)
        report = load_model(model, args.model, strict=True)
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        payload["coverage"] = {
            "status": "passed",
            "loaded": len(report.loaded),
            "missing": len(report.missing),
            "duplicate": len(report.duplicate),
            "unexpected_text_weights": len(report.unexpected_text_weights),
            "intentionally_skipped_non_text": len(
                report.intentionally_skipped_non_text
            ),
            "tied_aliases": len(report.tied_aliases),
            "parameter_bytes": parameter_bytes,
            "missing_names": sorted(report.missing),
            "duplicate_names": sorted(report.duplicate),
            "unexpected_text_weight_names": sorted(
                report.unexpected_text_weights
            ),
        }
    except WeightLoadingError as exc:
        report = exc.report
        payload["coverage"] = {
            "status": "failed",
            "loaded": len(report.loaded),
            "missing": len(report.missing),
            "duplicate": len(report.duplicate),
            "unexpected_text_weights": len(report.unexpected_text_weights),
            "intentionally_skipped_non_text": len(
                report.intentionally_skipped_non_text
            ),
            "tied_aliases": len(report.tied_aliases),
            "missing_names": sorted(report.missing),
            "duplicate_names": sorted(report.duplicate),
            "unexpected_text_weight_names": sorted(
                report.unexpected_text_weights
            ),
            "error": safe_error(exc),
        }
        write_outputs(args, payload)
        raise
    finally:
        torch.set_default_dtype(previous_dtype)

    write_outputs(args, payload)


def write_outputs(args, payload):
    coverage = payload["coverage"]
    write_json(args.save_json, payload)
    table = markdown_table(
        ["field", "value"],
        [[key, value] for key, value in coverage.items() if not isinstance(value, (list, dict))],
    )
    details = []
    for key in ("missing_names", "duplicate_names", "unexpected_text_weight_names"):
        if coverage.get(key):
            details.append(f"## {key}\n\n" + "\n".join(f"- `{name}`" for name in coverage[key]))
    write_text(
        args.save_md,
        "# Qwen3.5 Text Checkpoint Coverage\n\n"
        + table
        + ("\n\n" + "\n\n".join(details) if details else ""),
    )
    print(f"Saved {args.save_json}", flush=True)
    print(f"Saved {args.save_md}", flush=True)


if __name__ == "__main__":
    main()
