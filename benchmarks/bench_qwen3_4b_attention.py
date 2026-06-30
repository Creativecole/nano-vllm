from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_attention_decode import markdown_table, parse_int_list, run_case
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
    args = parser.parse_args()

    shapes = load_qwen_attention_shapes(args.model)
    rows = []
    for backend in [item.strip() for item in args.attn_backends.split(",") if item.strip()]:
        case_args = argparse.Namespace(**vars(args))
        case_args.backend = backend
        for batch_size in parse_int_list(args.batch_sizes):
            for seq_len in parse_int_list(args.seq_lens):
                rows.append(run_case(case_args, shapes, batch_size, seq_len, args.block_size))

    payload = {
        "model": args.model,
        "shapes": shapes.to_dict(),
        "rows": rows,
    }
    if args.save_json:
        path = Path(args.save_json)
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
    print(md)
    if args.save_md:
        path = Path(args.save_md)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()

