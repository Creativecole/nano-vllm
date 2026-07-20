#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.qwen35_hybrid.common import (  # noqa: E402
    DEFAULT_RESULTS_DIR,
    deterministic_prompts,
    environment_metadata,
    error_metrics,
    load_hf_text_reference,
    load_model_facts,
    load_nano_text_reference,
    markdown_table,
    parse_int_list,
    require_cuda,
    safe_error,
    write_json,
    write_text,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare real Qwen3.5 HF, nano no-cache, and hybrid serving outputs."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-lens", default="1,16,128,512")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--decode-steps", default="1,8,32")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument(
        "--skip-continuous-batching",
        action="store_true",
        help="Skip the mixed-length dynamic-admission workload for staged spot checks.",
    )
    parser.add_argument(
        "--save-json",
        default=str(DEFAULT_RESULTS_DIR / "correctness.json"),
    )
    parser.add_argument(
        "--save-md",
        default=str(REPO_ROOT / "docs/qwen35_hybrid/03_correctness.md"),
    )
    parser.add_argument(
        "--keep-worker-artifacts",
        action="store_true",
        help="Keep intermediate tensor artifacts used for cross-process comparison.",
    )
    parser.add_argument("--worker", choices=("hf", "nano_no_cache", "nano_serving"))
    parser.add_argument("--artifact")
    return parser.parse_args()


def case_key(batch_size: int, prompt_len: int, decode_steps: int) -> str:
    return f"b{batch_size}_p{prompt_len}_d{decode_steps}"


def logit_key(prompt_len: int, step: int) -> str:
    return f"b1_p{prompt_len}_step{step}"


def continuous_workload(vocab_size: int):
    prompt_lens = (16, 128, 512)
    decode_steps = (8, 32, 16)
    prompts = [
        deterministic_prompts(vocab_size, 1, length, seed=101 + row)[0]
        for row, length in enumerate(prompt_lens)
    ]
    return list(zip(prompts, decode_steps))


def install_layer_hooks(model, destination):
    handles = []
    for layer_idx, layer in enumerate(model.model.layers):
        def capture(_module, _inputs, output, index=layer_idx):
            tensor = output[0] if isinstance(output, tuple) else output
            destination[index] = tensor[:, -1].float().cpu()

        handles.append(layer.register_forward_hook(capture))
    return handles


def report_to_dict(report):
    return {
        "loaded": len(report.loaded),
        "missing": len(report.missing),
        "duplicate": len(report.duplicate),
        "unexpected_text_weights": len(report.unexpected_text_weights),
        "intentionally_skipped_non_text": len(report.intentionally_skipped_non_text),
        "tied_aliases": len(report.tied_aliases),
    }


@torch.inference_mode()
def run_hf_worker(args, prompt_lens, batch_sizes, decode_steps):
    require_cuda()
    print("[correctness:hf] loading text-only reference", flush=True)
    model, config, report = load_hf_text_reference(args.model)
    payload = {
        "backend": "huggingface_eager",
        "coverage": report_to_dict(report),
        "prefill_layers": {},
        "prefill_logits": {},
        "decode_logits": {},
        "tokens": {},
    }
    for prompt_len in prompt_lens:
        print(f"[correctness:hf] no-cache prompt={prompt_len}", flush=True)
        prompts = deterministic_prompts(config.vocab_size, 1, prompt_len)
        captured = {}
        handles = install_layer_hooks(model, captured)
        outputs = model(
            input_ids=torch.tensor(prompts, device="cuda"),
            use_cache=False,
        )
        for handle in handles:
            handle.remove()
        payload["prefill_layers"][str(prompt_len)] = captured
        payload["prefill_logits"][str(prompt_len)] = outputs.logits[:, -1].float().cpu()

    max_steps = max(decode_steps)
    selected_steps = set(decode_steps)
    for batch_size in batch_sizes:
        for prompt_len in prompt_lens:
            print(
                f"[correctness:hf] cached batch={batch_size} prompt={prompt_len} "
                f"steps={max_steps}",
                flush=True,
            )
            prompts = deterministic_prompts(
                config.vocab_size, batch_size, prompt_len
            )
            input_ids = torch.tensor(prompts, device="cuda")
            outputs = model(input_ids=input_ids, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1]
            generated = [[] for _ in range(batch_size)]
            for step in range(1, max_steps + 1):
                tokens = logits.argmax(dim=-1)
                for row, token in enumerate(tokens.tolist()):
                    generated[row].append(token)
                if batch_size == 1 and step in selected_steps:
                    payload["decode_logits"][logit_key(prompt_len, step)] = (
                        logits.float().cpu()
                    )
                if step != max_steps:
                    outputs = model(
                        input_ids=tokens.unsqueeze(1),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = outputs.past_key_values
                    logits = outputs.logits[:, -1]
            for steps in decode_steps:
                payload["tokens"][case_key(batch_size, prompt_len, steps)] = [
                    row[:steps] for row in generated
                ]
            del cache, outputs, logits
    payload["continuous_tokens"] = {}
    if not args.skip_continuous_batching:
        for row, (prompt, steps) in enumerate(continuous_workload(config.vocab_size)):
            input_ids = torch.tensor([prompt], device="cuda")
            outputs = model(input_ids=input_ids, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1]
            tokens = []
            for step in range(steps):
                token = logits.argmax(-1)
                tokens.append(int(token))
                if step + 1 < steps:
                    outputs = model(
                        input_ids=token.unsqueeze(1),
                        past_key_values=cache,
                        use_cache=True,
                    )
                    cache = outputs.past_key_values
                    logits = outputs.logits[:, -1]
            payload["continuous_tokens"][str(row)] = tokens
    return payload


@torch.inference_mode()
def run_nano_no_cache_worker(args, prompt_lens):
    require_cuda()
    print("[correctness:nano-no-cache] loading text-only reference", flush=True)
    model, config, report = load_nano_text_reference(args.model)
    payload = {
        "backend": "nanovllm_no_cache",
        "coverage": report_to_dict(report),
        "prefill_layers": {},
        "prefill_logits": {},
    }
    for prompt_len in prompt_lens:
        print(
            f"[correctness:nano-no-cache] prompt={prompt_len}", flush=True
        )
        prompts = deterministic_prompts(config.vocab_size, 1, prompt_len)
        captured = {}
        handles = install_layer_hooks(model, captured)
        hidden = model(torch.tensor(prompts, device="cuda"))
        logits = model.compute_logits(hidden)
        for handle in handles:
            handle.remove()
        payload["prefill_layers"][str(prompt_len)] = captured
        payload["prefill_logits"][str(prompt_len)] = logits[:, -1].float().cpu()
    return payload


@torch.inference_mode()
def run_nano_serving_worker(args, prompt_lens, batch_sizes, decode_steps):
    require_cuda()
    from nanovllm import LLM, SamplingParams

    facts = load_model_facts(args.model)
    max_batch = max(batch_sizes)
    max_steps = max(decode_steps)
    print("[correctness:nano-serving] loading hybrid engine", flush=True)
    llm = LLM(
        args.model,
        enforce_eager=True,
        max_num_seqs=max_batch,
        hybrid_state_capacity=max_batch,
        max_model_len=max(prompt_lens) + max_steps + 1,
        max_num_batched_tokens=max_batch * max(prompt_lens),
    )
    payload = {
        "backend": "nanovllm_hybrid",
        "decode_logits": {},
        "tokens": {},
        "hybrid_cache": llm.model_runner.call("get_hybrid_state_stats"),
        "continuous_tokens": {},
    }
    selected_steps = set(decode_steps)
    try:
        for batch_size in batch_sizes:
            for prompt_len in prompt_lens:
                print(
                    f"[correctness:nano-serving] batch={batch_size} "
                    f"prompt={prompt_len} steps={max_steps}",
                    flush=True,
                )
                prompts = deterministic_prompts(
                    int(facts["vocab_size"]), batch_size, prompt_len
                )
                params = SamplingParams(
                    temperature=0.0,
                    max_tokens=max_steps,
                    ignore_eos=True,
                )
                seq_ids = [llm.add_request(prompt, params) for prompt in prompts]
                generated = {seq_id: [] for seq_id in seq_ids}
                while not llm.is_finished():
                    _, _, step_logits = llm.step_with_logits()
                    for seq_id, logits in step_logits.items():
                        token = int(logits.argmax())
                        generated[seq_id].append(token)
                        step = len(generated[seq_id])
                        if batch_size == 1 and step in selected_steps:
                            payload["decode_logits"][logit_key(prompt_len, step)] = (
                                logits.unsqueeze(0)
                            )
                rows = [generated[seq_id] for seq_id in seq_ids]
                for steps in decode_steps:
                    payload["tokens"][case_key(batch_size, prompt_len, steps)] = [
                        row[:steps] for row in rows
                    ]

        if not args.skip_continuous_batching:
            workload = continuous_workload(int(facts["vocab_size"]))
            first_ids = [
                llm.add_request(
                    prompt,
                    SamplingParams(
                        temperature=0.0, max_tokens=steps, ignore_eos=True
                    ),
                )
                for prompt, steps in workload[:2]
            ]
            generated = {seq_id: [] for seq_id in first_ids}
            _, _, step_logits = llm.step_with_logits()
            for seq_id, logits in step_logits.items():
                generated[seq_id].append(int(logits.argmax()))
            third_prompt, third_steps = workload[2]
            third_id = llm.add_request(
                third_prompt,
                SamplingParams(
                    temperature=0.0,
                    max_tokens=third_steps,
                    ignore_eos=True,
                ),
            )
            generated[third_id] = []
            while not llm.is_finished():
                _, _, step_logits = llm.step_with_logits()
                for seq_id, logits in step_logits.items():
                    generated[seq_id].append(int(logits.argmax()))
            for row, seq_id in enumerate([*first_ids, third_id]):
                payload["continuous_tokens"][str(row)] = generated[seq_id]
    finally:
        llm.exit()
    return payload


def run_worker(args, prompt_lens, batch_sizes, decode_steps):
    try:
        if args.worker == "hf":
            payload = run_hf_worker(args, prompt_lens, batch_sizes, decode_steps)
        elif args.worker == "nano_no_cache":
            payload = run_nano_no_cache_worker(args, prompt_lens)
        else:
            payload = run_nano_serving_worker(
                args, prompt_lens, batch_sizes, decode_steps
            )
        torch.save(payload, args.artifact)
    except Exception as exc:
        torch.save({"backend": args.worker, "error": safe_error(exc)}, args.artifact)
        raise


def invoke_worker(args, worker: str, artifact: Path):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--prompt-lens",
        args.prompt_lens,
        "--batch-sizes",
        args.batch_sizes,
        "--decode-steps",
        args.decode_steps,
        "--worker",
        worker,
        "--artifact",
        str(artifact),
    ]
    if args.skip_continuous_batching:
        command.append("--skip-continuous-batching")
    print(f"[correctness] starting {worker}", flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get(
        "PYTHONPATH", ""
    )
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
    print(f"[correctness] finished {worker}", flush=True)
    return torch.load(artifact, map_location="cpu", weights_only=False)


def compare_payloads(args, hf, nano_no_cache, nano_serving, prompt_lens):
    prefill_rows = []
    layer_rows = []
    first_mismatch = None
    for prompt_len in prompt_lens:
        key = str(prompt_len)
        metrics = error_metrics(
            nano_no_cache["prefill_logits"][key], hf["prefill_logits"][key]
        )
        greedy_match = torch.equal(
            nano_no_cache["prefill_logits"][key].argmax(-1),
            hf["prefill_logits"][key].argmax(-1),
        )
        prefill_rows.append(
            {"prompt_len": prompt_len, "greedy_match": greedy_match, **metrics}
        )
        for layer_idx in sorted(hf["prefill_layers"][key]):
            expected = hf["prefill_layers"][key][layer_idx]
            actual = nano_no_cache["prefill_layers"][key][layer_idx]
            layer_metrics = error_metrics(actual, expected)
            close = torch.isclose(
                actual.float(), expected.float(), rtol=args.rtol, atol=args.atol
            )
            row = {
                "prompt_len": prompt_len,
                "layer_idx": layer_idx,
                "close_fraction": close.float().mean().item(),
                **layer_metrics,
            }
            layer_rows.append(row)
            if first_mismatch is None and not bool(close.all()):
                first_mismatch = row

    decode_rows = []
    for key in sorted(hf["decode_logits"]):
        metrics = error_metrics(
            nano_serving["decode_logits"][key], hf["decode_logits"][key]
        )
        decode_rows.append(
            {
                "case": key,
                "greedy_match": torch.equal(
                    nano_serving["decode_logits"][key].argmax(-1),
                    hf["decode_logits"][key].argmax(-1),
                ),
                **metrics,
            }
        )

    token_rows = []
    matched_tokens = total_tokens = 0
    for key in sorted(hf["tokens"]):
        expected_rows = hf["tokens"][key]
        actual_rows = nano_serving["tokens"][key]
        matches = sum(
            int(actual == expected)
            for actual_row, expected_row in zip(actual_rows, expected_rows)
            for actual, expected in zip(actual_row, expected_row)
        )
        count = sum(len(row) for row in expected_rows)
        matched_tokens += matches
        total_tokens += count
        token_rows.append(
            {
                "case": key,
                "matched_tokens": matches,
                "total_tokens": count,
                "token_match_rate": matches / count if count else 1.0,
            }
        )
    continuous_rows = []
    for key in sorted(hf["continuous_tokens"]):
        expected = hf["continuous_tokens"][key]
        actual = nano_serving["continuous_tokens"][key]
        matches = sum(int(left == right) for left, right in zip(actual, expected))
        matched_tokens += matches
        total_tokens += len(expected)
        continuous_rows.append(
            {
                "request": key,
                "matched_tokens": matches,
                "total_tokens": len(expected),
                "token_match_rate": matches / len(expected),
            }
        )
    return {
        "prefill_logits": prefill_rows,
        "layer_comparison": layer_rows,
        "first_mismatched_layer": first_mismatch,
        "decode_logits": decode_rows,
        "greedy_tokens": token_rows,
        "continuous_batching": continuous_rows,
        "token_match_rate": matched_tokens / total_tokens if total_tokens else 1.0,
        "matched_tokens": matched_tokens,
        "total_tokens": total_tokens,
    }


def render_markdown(payload):
    comparison = payload["comparison"]
    prefill = markdown_table(
        ["prompt", "greedy", "max_abs", "mean_abs", "max_rel", "mean_rel"],
        [
            [
                row["prompt_len"],
                row["greedy_match"],
                row["max_abs_error"],
                row["mean_abs_error"],
                row["max_relative_error"],
                row["mean_relative_error"],
            ]
            for row in comparison["prefill_logits"]
        ],
    )
    decode = markdown_table(
        ["case", "greedy", "max_abs", "mean_abs", "max_rel", "mean_rel"],
        [
            [
                row["case"],
                row["greedy_match"],
                row["max_abs_error"],
                row["mean_abs_error"],
                row["max_relative_error"],
                row["mean_relative_error"],
            ]
            for row in comparison["decode_logits"]
        ],
    )
    tokens = markdown_table(
        ["case", "matched", "total", "match rate"],
        [
            [
                row["case"],
                row["matched_tokens"],
                row["total_tokens"],
                row["token_match_rate"],
            ]
            for row in comparison["greedy_tokens"]
        ],
    )
    continuous = markdown_table(
        ["request", "matched", "total", "match rate"],
        [
            [
                row["request"],
                row["matched_tokens"],
                row["total_tokens"],
                row["token_match_rate"],
            ]
            for row in comparison["continuous_batching"]
        ],
    )
    mismatch = comparison["first_mismatched_layer"]
    mismatch_text = "None at configured tolerances." if mismatch is None else str(mismatch)
    return f"""# Qwen3.5 Hybrid Correctness

This report compares the official Hugging Face eager text model, nano-vLLM's
no-cache reference, and nano-vLLM's real Scheduler + paged-KV + DeltaNet serving path.

## Summary

- Greedy token match rate: `{comparison['token_match_rate']:.6f}`
- Matched tokens: `{comparison['matched_tokens']} / {comparison['total_tokens']}`
- First mismatched layer: `{mismatch_text}`

Maximum relative error can be large around reference values close to zero; max/mean
absolute error and greedy-token agreement should be read together.

## Prefill Final Logits

{prefill}

## Selected Decode Logits

{decode}

## Multi-step Greedy Tokens

{tokens}

## Dynamic Continuous Batching

Two requests start together and a third request is admitted after the first engine
step. Prompt lengths and generation lengths differ, so this also exercises completion
and active-batch compaction.

{continuous}
"""


def main():
    args = parse_args()
    prompt_lens = parse_int_list(args.prompt_lens)
    batch_sizes = parse_int_list(args.batch_sizes)
    decode_steps = parse_int_list(args.decode_steps)
    if args.worker:
        if not args.artifact:
            raise ValueError("--artifact is required in worker mode")
        run_worker(args, prompt_lens, batch_sizes, decode_steps)
        return

    require_cuda()
    artifact_dir = DEFAULT_RESULTS_DIR / ".correctness_workers"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        name: artifact_dir / f"{name}.pt"
        for name in ("hf", "nano_no_cache", "nano_serving")
    }
    try:
        workers = {
            name: invoke_worker(args, name, artifact)
            for name, artifact in artifacts.items()
        }
        comparison = compare_payloads(
            args,
            workers["hf"],
            workers["nano_no_cache"],
            workers["nano_serving"],
            prompt_lens,
        )
        payload = {
            "environment": environment_metadata(args.model),
            "model_facts": load_model_facts(args.model),
            "matrix": {
                "prompt_lens": prompt_lens,
                "batch_sizes": batch_sizes,
                "decode_steps": decode_steps,
                "atol": args.atol,
                "rtol": args.rtol,
            },
            "weight_coverage": {
                name: worker.get("coverage")
                for name, worker in workers.items()
                if worker.get("coverage") is not None
            },
            "comparison": comparison,
        }
        write_json(args.save_json, payload)
        write_text(args.save_md, render_markdown(payload))
        print(f"Saved {args.save_json}", flush=True)
        print(f"Saved {args.save_md}", flush=True)
        if comparison["token_match_rate"] != 1.0:
            raise SystemExit("Greedy token mismatch detected; inspect correctness report")
    finally:
        if not args.keep_worker_artifacts:
            for artifact in artifacts.values():
                artifact.unlink(missing_ok=True)
            try:
                artifact_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    main()
