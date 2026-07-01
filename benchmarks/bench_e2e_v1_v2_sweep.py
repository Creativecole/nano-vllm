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
    "prompt_len",
    "block_size",
    "decode_tokens_per_s",
    "itl_ms_avg",
    "decode_step_ms_p50",
    "decode_step_ms_p95",
]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def summarize_case(backend: str, prompt_len: int, block_size: int, rows: list[dict]) -> dict:
    return {
        "backend": backend,
        "prompt_len": prompt_len,
        "block_size": block_size,
        "decode_tokens_per_s": mean_metric(rows, "decode_tokens_per_s"),
        "itl_ms_avg": mean_metric(rows, "itl_ms_avg"),
        "decode_step_ms_p50": mean_metric(rows, "decode_step_ms_p50"),
        "decode_step_ms_p95": mean_metric(rows, "decode_step_ms_p95"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep v1/v2 E2E decode over prompt length and KV block size.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--backends", default="triton_paged_decode,triton_paged_decode_v2")
    parser.add_argument("--prompt-lens", default="512,1024,2048,4096,8192")
    parser.add_argument("--block-sizes", default="16,32,64,128,256")
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", dest="max_tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--auto-threshold", type=int, default=1024)
    parser.add_argument("--save-md", default="results/rtx5090_qwen3_4b/e2e_v1_v2_prompt_block_sweep.md")
    parser.add_argument("--save-json", default="results/rtx5090_qwen3_4b/e2e_v1_v2_prompt_block_sweep.json")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_e2e_v1_v2_sweep.py")

    from nanovllm import LLM, SamplingParams

    summary_rows = []
    all_runs = []
    for block_size in parse_int_list(args.block_sizes):
        for prompt_len in parse_int_list(args.prompt_lens):
            for backend in parse_str_list(args.backends):
                case_args = argparse.Namespace(**vars(args))
                case_args.prompt_len = prompt_len
                case_args.block_size = block_size
                case_args.attn_backend = backend
                print(
                    f"[sweep] backend={backend} prompt_len={prompt_len} block_size={block_size}",
                    flush=True,
                )
                for warmup_idx in range(args.warmup):
                    print(f"[sweep] warmup {warmup_idx + 1}/{args.warmup}", flush=True)
                    run_once(case_args, warmup_idx + 1, LLM, SamplingParams)
                rows = []
                for run_idx in range(args.repeat):
                    print(f"[sweep] run {run_idx + 1}/{args.repeat}", flush=True)
                    rows.append(run_once(case_args, run_idx + 1, LLM, SamplingParams))
                summary_rows.append(summarize_case(backend, prompt_len, block_size, rows))
                all_runs.extend(rows)

    payload = {
        "benchmark": "e2e_v1_v2_prompt_block_sweep",
        "model": args.model,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "summary": summary_rows,
        "runs": all_runs,
    }
    md = "\n".join([
        "# E2E v1/v2 Prompt/Block Sweep",
        "",
        f"- model: `{args.model}`",
        f"- prompt_lens: `{args.prompt_lens}`",
        f"- block_sizes: `{args.block_sizes}`",
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[save] JSON {path}", flush=True)
    if args.save_md:
        path = Path(args.save_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
        print(f"[save] Markdown {path}", flush=True)


if __name__ == "__main__":
    main()
