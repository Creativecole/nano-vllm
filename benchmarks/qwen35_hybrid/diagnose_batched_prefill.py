#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    deterministic_prompts,
    environment_metadata,
    load_model_facts,
    markdown_table,
    peak_memory_gb,
    require_cuda,
    reset_peak_memory,
    synchronize,
    write_json,
    write_text,
)


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        try:
            batch, prompt = (int(part) for part in item.split("x", maxsplit=1))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid case {item!r}; expected comma-separated BxT values"
            ) from exc
        if batch <= 0 or prompt <= 0:
            raise ValueError(f"Case dimensions must be positive, got {item!r}")
        cases.append((batch, prompt))
    if not cases:
        raise ValueError("At least one diagnostic case is required")
    return cases


def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose Qwen3.5 batched chunked DeltaNet serving prefill."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", default="1x128,4x128,4x512")
    parser.add_argument("--deltanet-chunk-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "diagnose_batched_prefill.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/08_batched_prefill_diagnosis.md"),
    )
    return parser.parse_args()


def add_requests(llm, facts, batch_size, prompt_len):
    from nanovllm import SamplingParams

    prompts = deterministic_prompts(
        int(facts["vocab_size"]), batch_size, prompt_len
    )
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    for prompt in prompts:
        llm.add_request(prompt, params)


def run_prefill(llm, facts, batch_size, prompt_len):
    add_requests(llm, facts, batch_size, prompt_len)
    _, num_tokens = llm.step()
    if num_tokens <= 0:
        raise RuntimeError(
            f"Expected a prefill step for batch={batch_size}, prompt={prompt_len}"
        )
    while not llm.is_finished():
        llm.step()


def summarize_case(batch_size, prompt_len, elapsed_s, peak_gb, diagnostics):
    layer_stats = diagnostics["layers"]
    recurrence_counts = []
    observed_batches = set()
    fast_path_layers = 0
    fallback_layers = 0
    layer_rows = []
    for layer_idx, stats in sorted(layer_stats.items(), key=lambda item: int(item[0])):
        calls = stats["calls"]
        recurrence_counts.append(stats["recurrence_calls"])
        call_batches = sorted(
            {
                int(call["query_shape"][0])
                for call in calls
                if call["is_prefill"] and call["backend"] == "chunked"
            }
        )
        observed_batches.update(call_batches)
        fast_path_layers += int(stats["equal_length_batched_prefill_calls"] > 0)
        fallback_layers += int(stats["variable_length_fallback_calls"] > 0)
        layer_rows.append(
            {
                "layer_id": int(layer_idx),
                "recurrence_calls": stats["recurrence_calls"],
                "observed_batch_dimensions": call_batches,
                "equal_length_batched_prefill_calls": stats[
                    "equal_length_batched_prefill_calls"
                ],
                "variable_length_fallback_calls": stats[
                    "variable_length_fallback_calls"
                ],
                "fallback_sequences": stats["fallback_sequences"],
                "calls": calls,
            }
        )

    state = diagnostics["state_manager"]
    return {
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "prefill_time_ms": elapsed_s * 1000,
        "peak_memory_gb": peak_gb,
        "deltanet_layers": len(layer_rows),
        "recurrence_calls_total": sum(recurrence_counts),
        "recurrence_calls_per_layer_min": min(recurrence_counts),
        "recurrence_calls_per_layer_max": max(recurrence_counts),
        "observed_recurrence_batch_dimensions": sorted(observed_batches),
        "equal_length_fast_path_layers": fast_path_layers,
        "variable_length_fallback_layers": fallback_layers,
        "all_layers_single_batched_call": all(
            count == 1 for count in recurrence_counts
        )
        and observed_batches == {batch_size},
        "state_gather_calls": state["gather_calls"],
        "state_gather_layer_ops": state["gather_layer_ops"],
        "state_gather_bytes": state["gather_bytes"],
        "state_commit_calls": state["commit_calls"],
        "state_commit_layer_ops": state["commit_layer_ops"],
        "state_commit_bytes": state["commit_bytes"],
        "slot_id_upload_bytes": state["slot_id_upload_bytes"],
        "layers": layer_rows,
    }


def render_markdown(payload):
    table = markdown_table(
        [
            "batch",
            "prompt",
            "prefill ms",
            "calls/layer min-max",
            "observed B",
            "single batched call",
            "gather calls/layer ops",
            "gather MiB",
            "commit calls/layer ops",
            "commit MiB",
            "peak GiB",
        ],
        [
            [
                row["batch_size"],
                row["prompt_len"],
                row["prefill_time_ms"],
                f"{row['recurrence_calls_per_layer_min']}-"
                f"{row['recurrence_calls_per_layer_max']}",
                row["observed_recurrence_batch_dimensions"],
                row["all_layers_single_batched_call"],
                f"{row['state_gather_calls']}/"
                f"{row['state_gather_layer_ops']}",
                row["state_gather_bytes"] / 2**20,
                f"{row['state_commit_calls']}/"
                f"{row['state_commit_layer_ops']}",
                row["state_commit_bytes"] / 2**20,
                row["peak_memory_gb"],
            ]
            for row in payload["cases"]
        ],
    )
    return f"""# Qwen3.5 Batched Chunked Prefill Diagnosis

The previous packed path split hidden states by request in
`Qwen3_5GatedDeltaNet._forward_packed` and called recurrence once per sequence per
DeltaNet layer. The equal-length chunked fast path reshapes packed tokens to
`[batch, sequence, hidden]` and calls recurrence once per layer.

{table}

`HybridStateManager` gathers and commits all active request slots with one batched
`index_select` / `index_copy_` operation per DeltaNet layer. The byte counters report
the state payload moved by those layer operations; they do not imply a per-request
Python copy loop.

The full JSON includes every recurrence Q/K/V/state shape. For an equal-length batch-4
case, acceptance requires one recurrence call per DeltaNet layer and a recorded query
batch dimension of 4. Variable-length packed requests intentionally retain the
correctness-first per-sequence fallback.
"""


@torch.inference_mode()
def main():
    args = parse_args()
    require_cuda()
    cases = parse_cases(args.cases)
    if args.warmup < 0 or args.deltanet_chunk_size <= 0:
        raise ValueError("warmup must be non-negative and chunk size must be positive")

    from nanovllm import LLM

    facts = load_model_facts(args.model)
    max_batch = max(batch for batch, _ in cases)
    max_prompt = max(prompt for _, prompt in cases)
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=max_batch,
        hybrid_state_capacity=max_batch,
        max_model_len=max_prompt + 2,
        max_num_batched_tokens=max_batch * max_prompt,
        deltanet_backend="chunked",
        deltanet_chunk_size=args.deltanet_chunk_size,
    )
    rows = []
    try:
        for batch_size, prompt_len in cases:
            print(
                f"[diagnose] batch={batch_size} prompt={prompt_len} warmup",
                flush=True,
            )
            llm.model_runner.call("set_deltanet_diagnostics", False, True)
            for _ in range(args.warmup):
                run_prefill(llm, facts, batch_size, prompt_len)

            llm.model_runner.call("set_deltanet_diagnostics", True, True)
            reset_peak_memory()
            synchronize()
            start = time.perf_counter()
            add_requests(llm, facts, batch_size, prompt_len)
            _, num_tokens = llm.step()
            synchronize()
            elapsed_s = time.perf_counter() - start
            if num_tokens <= 0:
                raise RuntimeError("Diagnostic measurement did not execute prefill")
            diagnostics = llm.model_runner.call("get_deltanet_diagnostics")
            llm.model_runner.call("set_deltanet_diagnostics", False, False)
            while not llm.is_finished():
                llm.step()
            row = summarize_case(
                batch_size,
                prompt_len,
                elapsed_s,
                peak_memory_gb(),
                diagnostics,
            )
            rows.append(row)
            print(
                f"[diagnose] calls/layer="
                f"{row['recurrence_calls_per_layer_min']}-"
                f"{row['recurrence_calls_per_layer_max']} "
                f"observed_batch={row['observed_recurrence_batch_dimensions']}",
                flush=True,
            )
    finally:
        llm.exit()

    payload = {
        "environment": environment_metadata(args.model),
        "model_facts": facts,
        "matrix": {
            "cases": [list(case) for case in cases],
            "deltanet_backend": "chunked",
            "deltanet_chunk_size": args.deltanet_chunk_size,
            "warmup": args.warmup,
        },
        "python_loop_root_cause": (
            "Qwen3_5GatedDeltaNet._forward_packed variable-length fallback"
        ),
        "cases": rows,
    }
    write_json(args.save_json, payload)
    write_text(args.save_md, render_markdown(payload))
    print(f"Saved {args.save_json}", flush=True)
    print(f"Saved {args.save_md}", flush=True)


if __name__ == "__main__":
    main()
