from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_attention_decode import describe_case, log, markdown_table, parse_int_list, run_case, verbose_case_details
from nanovllm.utils.shapes import load_qwen_attention_shapes


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 attention backend summary benchmark.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attn-backends", default="torch_paged,triton_paged_decode")
    parser.add_argument("--seq-lens", default="1024,4096,8192")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--save-md")
    parser.add_argument("--save-json")
    parser.add_argument("--verbose", action="store_true", help="Print tensor shapes and model attention dimensions.")
    args = parser.parse_args()

    log(f"[setup] loading model shapes from {args.model}")
    shapes = load_qwen_attention_shapes(args.model)
    verbose_case_details(args, shapes)
    rows = []
    for backend in [item.strip() for item in args.attn_backends.split(",") if item.strip()]:
        log(f"[backend] start backend={backend}")
        case_args = argparse.Namespace(**vars(args))
        case_args.backend = backend
        for batch_size in parse_int_list(args.batch_sizes):
            for seq_len in parse_int_list(args.seq_lens):
                try:
                    log(
                        "[backend] scheduling "
                        f"{describe_case(case_args, shapes, batch_size, seq_len, args.block_size)}"
                    )
                    rows.append(run_case(case_args, shapes, batch_size, seq_len, args.block_size))
                except Exception as exc:
                    log(
                        "[error] benchmark failed for "
                        f"{describe_case(case_args, shapes, batch_size, seq_len, args.block_size)}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise
        log(f"[backend] done backend={backend}")

    payload = {
        "model": args.model,
        "shapes": shapes.to_dict(),
        "rows": rows,
    }
    if args.save_json:
        path = Path(args.save_json)
        log(f"[save] writing JSON results to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md = "\n".join([
        "# Qwen3 Attention Backend Summary",
        "",
        f"- model: `{args.model}`",
        f"- dtype: `{args.dtype}`",
        f"- q_heads / kv_heads / head_dim: `{shapes.num_attention_heads}` / `{shapes.num_key_value_heads}` / `{shapes.head_dim}`",
        f"- GQA ratio: `{shapes.gqa_ratio}`",
        "",
        markdown_table(rows),
        "",
        "This is the main attention-backend benchmark artifact. It isolates paged decode attention and does not claim end-to-end speedup.",
        "",
    ])
    log("[summary] benchmark table")
    print(md, flush=True)
    if args.save_md:
        path = Path(args.save_md)
        log(f"[save] writing Markdown results to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")
    if not args.save_json and not args.save_md:
        log("[save] no output path requested; results printed to stdout only")


if __name__ == "__main__":
    main()
