from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench_e2e import format_value, run_once


SUMMARY_COLUMNS = [
    "backend",
    "elapsed_s",
    "ttft_s",
    "decode_time_s",
    "decode_tokens_per_s",
    "itl_ms_avg",
    "decode_step_ms_p50",
    "decode_step_ms_p95",
    "peak_gpu_memory_gb",
]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def markdown_rows(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def mean_metric(rows: list[dict], key: str) -> float:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    return mean(values) if values else 0.0


def summarize_backend(backend: str, rows: list[dict]) -> dict:
    return {
        "backend": backend,
        "elapsed_s": mean_metric(rows, "elapsed_s"),
        "ttft_s": mean_metric(rows, "ttft_s"),
        "decode_time_s": mean_metric(rows, "decode_time_s"),
        "decode_tokens_per_s": mean_metric(rows, "decode_tokens_per_s"),
        "itl_ms_avg": mean_metric(rows, "itl_ms_avg"),
        "decode_step_ms_p50": mean_metric(rows, "decode_step_ms_p50"),
        "decode_step_ms_p95": mean_metric(rows, "decode_step_ms_p95"),
        "peak_gpu_memory_gb": mean_metric(rows, "peak_gpu_memory_gb"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare E2E attention backends on the same workload.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backends", default="flash_attn,triton_paged_decode,triton_paged_decode_v2")
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", dest="max_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--save-md", default="results/rtx5090_qwen3_4b/e2e_attention_backend_compare.md")
    parser.add_argument("--save-json", default="results/rtx5090_qwen3_4b/e2e_attention_backend_compare.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_e2e_attention_compare.py")

    from nanovllm import LLM, SamplingParams

    all_runs = {}
    summary_rows = []
    for backend in parse_str_list(args.backends):
        print(f"[backend] start backend={backend}", flush=True)
        backend_args = argparse.Namespace(**vars(args))
        backend_args.attn_backend = backend
        for warmup_idx in range(args.warmup):
            print(f"[backend] warmup backend={backend} {warmup_idx + 1}/{args.warmup}", flush=True)
            run_once(backend_args, warmup_idx + 1, LLM, SamplingParams)
        rows = []
        for run_idx in range(args.repeat):
            print(f"[backend] run backend={backend} {run_idx + 1}/{args.repeat}", flush=True)
            rows.append(run_once(backend_args, run_idx + 1, LLM, SamplingParams))
        all_runs[backend] = rows
        summary_rows.append(summarize_backend(backend, rows))
        print(f"[backend] done backend={backend}", flush=True)

    payload = {
        "benchmark": "e2e_attention_backend_compare",
        "model": args.model,
        "prompt_len": args.prompt_len,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "summary": summary_rows,
        "runs": all_runs,
    }
    md = "\n".join([
        "# E2E Attention Backend Compare",
        "",
        f"- model: `{args.model}`",
        f"- prompt_len: `{args.prompt_len}`",
        f"- num_prompts: `{args.num_prompts}`",
        f"- max_tokens: `{args.max_tokens}`",
        f"- repeat: `{args.repeat}`",
        "",
        markdown_rows(summary_rows, SUMMARY_COLUMNS),
        "",
    ])
    print(md, flush=True)
    if args.save_json:
        path = Path(args.save_json)
        print(f"[save] writing JSON results to {path}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    if args.save_md:
        path = Path(args.save_md)
        print(f"[save] writing Markdown results to {path}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
