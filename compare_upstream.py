import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean


RUNNER_CODE = r"""
import json
import sys
from time import perf_counter

payload = json.loads(sys.argv[1])
sys.path.insert(0, payload["repo"])

import torch


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def make_token_prompt(llm, prompt_len):
    token_ids = llm.tokenizer.encode(" benchmark", add_special_tokens=False)
    token_id = token_ids[0] if token_ids else 0
    eos = llm.tokenizer.eos_token_id
    if token_id == eos:
        token_id = 0 if eos != 0 else 1
    return [token_id] * prompt_len


def run_generation(llm, prompts, sampling_params):
    outputs = {}
    prefill_time_s = 0.0
    decode_time_s = 0.0
    prefill_tokens = 0
    decode_tokens = 0
    decode_latencies = []
    first_decode_end = None

    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    torch.cuda.synchronize()
    start = perf_counter()
    while not llm.is_finished():
        torch.cuda.synchronize()
        step_start = perf_counter()
        output, num_tokens = llm.step()
        torch.cuda.synchronize()
        step_elapsed = perf_counter() - step_start

        if num_tokens > 0:
            prefill_time_s += step_elapsed
            prefill_tokens += num_tokens
        else:
            step_decode_tokens = -num_tokens
            decode_time_s += step_elapsed
            decode_tokens += step_decode_tokens
            decode_latencies.append(step_elapsed)
            if first_decode_end is None:
                first_decode_end = perf_counter()

        for seq_id, token_ids in output:
            outputs[seq_id] = token_ids

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    generated_tokens = sum(len(token_ids) for token_ids in outputs.values())
    prompt_tokens = len(prompts[0]) * len(prompts) if prompts else 0
    return {
        "elapsed_s": elapsed,
        "ttft_s": (first_decode_end - start) if first_decode_end is not None else 0.0,
        "prefill_time_s": prefill_time_s,
        "decode_time_s": decode_time_s,
        "prompt_tokens": prompt_tokens,
        "prefill_tokens": prefill_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "total_tokens_per_s": (prompt_tokens + generated_tokens) / elapsed if elapsed else 0.0,
        "decode_tokens_per_s": decode_tokens / decode_time_s if decode_time_s else 0.0,
        "itl_ms_avg": decode_time_s / decode_tokens * 1000.0 if decode_tokens else 0.0,
        "decode_step_ms_p50": percentile(decode_latencies, 50) * 1000.0,
        "decode_step_ms_p95": percentile(decode_latencies, 95) * 1000.0,
    }


def run_once(run_index):
    from nanovllm import LLM, SamplingParams

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    llm = None
    try:
        llm = LLM(
            payload["model"],
            max_model_len=payload["prompt_len"] + payload["max_tokens"],
            max_num_seqs=payload["num_prompts"],
            enforce_eager=payload["enforce_eager"],
        )
        prompt = make_token_prompt(llm, payload["prompt_len"])
        prompts = [prompt[:] for _ in range(payload["num_prompts"])]
        sampling_params = SamplingParams(
            temperature=payload["temperature"],
            max_tokens=payload["max_tokens"],
        )
        row = run_generation(llm, prompts, sampling_params)
        row["run"] = run_index
        row["gpu"] = torch.cuda.get_device_name()
        row["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        metrics = llm.metrics() if hasattr(llm, "metrics") else {}
        row.update({
            "num_kvcache_blocks": metrics.get("num_kvcache_blocks", ""),
            "max_used_blocks": metrics.get("max_used_blocks", ""),
            "max_block_utilization": metrics.get("max_block_utilization", ""),
            "kv_cache_dtype": metrics.get("kv_cache_dtype", ""),
            "linear_backend": metrics.get("linear_backend", ""),
            "norm_backend": metrics.get("norm_backend", ""),
            "activation_backend": metrics.get("activation_backend", ""),
            "rope_backend": metrics.get("rope_backend", ""),
        })
        return row
    finally:
        if llm is not None:
            llm.exit()


if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")

for i in range(payload["warmup"]):
    run_once(-(i + 1))

rows = [run_once(i + 1) for i in range(payload["repeat"])]
result = {
    "repo": payload["repo"],
    "label": payload["label"],
    "model": payload["model"],
    "prompt_len": payload["prompt_len"],
    "num_prompts": payload["num_prompts"],
    "max_tokens": payload["max_tokens"],
    "repeat": payload["repeat"],
    "rows": rows,
}
print("JSON_RESULT_BEGIN")
print(json.dumps(result, sort_keys=True))
print("JSON_RESULT_END")
"""


def format_value(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[idx]


def markdown_rows(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def summarize(rows: list[dict]) -> dict:
    numeric_keys = [
        "elapsed_s",
        "ttft_s",
        "prefill_time_s",
        "decode_time_s",
        "total_tokens_per_s",
        "decode_tokens_per_s",
        "itl_ms_avg",
        "decode_step_ms_p50",
        "decode_step_ms_p95",
        "peak_gpu_memory_gb",
    ]
    summary = {}
    for key in numeric_keys:
        values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_p50"] = percentile(values, 50)
            summary[f"{key}_p95"] = percentile(values, 95)
    for key in [
        "gpu",
        "num_kvcache_blocks",
        "max_used_blocks",
        "max_block_utilization",
        "kv_cache_dtype",
        "linear_backend",
        "norm_backend",
        "activation_backend",
        "rope_backend",
    ]:
        if rows and key in rows[0]:
            summary[key] = rows[0][key]
    return summary


def run_repo(label: str, repo: Path, args) -> dict:
    payload = {
        "label": label,
        "repo": str(repo.resolve()),
        "model": args.model,
        "prompt_len": args.prompt_len,
        "num_prompts": args.num_prompts,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "enforce_eager": args.enforce_eager,
        "repeat": args.repeat,
        "warmup": args.warmup,
    }
    env = None
    completed = subprocess.run(
        [sys.executable, "-c", RUNNER_CODE, json.dumps(payload)],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "label": label,
            "repo": str(repo.resolve()),
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    match = re.search(r"JSON_RESULT_BEGIN\n(.*?)\nJSON_RESULT_END", completed.stdout, re.S)
    if not match:
        return {
            "label": label,
            "repo": str(repo.resolve()),
            "status": "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "error": "runner did not emit JSON_RESULT markers",
        }
    result = json.loads(match.group(1))
    result["status"] = "ok"
    result["summary"] = summarize(result["rows"])
    return result


def ratio_text(fork_value, upstream_value, higher_is_better: bool) -> str:
    if not isinstance(fork_value, (int, float)) or not isinstance(upstream_value, (int, float)) or upstream_value == 0:
        return ""
    ratio = fork_value / upstream_value
    if not higher_is_better and fork_value:
        ratio = upstream_value / fork_value
    return f"{ratio:.3f}x"


def build_report(results: list[dict], args) -> str:
    config_rows = [
        {"metric": "model", "value": args.model},
        {"metric": "prompt_len", "value": args.prompt_len},
        {"metric": "num_prompts", "value": args.num_prompts},
        {"metric": "max_tokens", "value": args.max_tokens},
        {"metric": "repeat", "value": args.repeat},
        {"metric": "warmup", "value": args.warmup},
        {"metric": "enforce_eager", "value": args.enforce_eager},
    ]

    ok_results = {result["label"]: result for result in results if result.get("status") == "ok"}
    comparison_rows = []
    metric_specs = [
        ("elapsed_s_mean", False),
        ("ttft_s_mean", False),
        ("decode_tokens_per_s_mean", True),
        ("itl_ms_avg_mean", False),
        ("decode_step_ms_p95_mean", False),
        ("peak_gpu_memory_gb_mean", False),
        ("max_used_blocks", False),
        ("max_block_utilization", False),
    ]
    fork = ok_results.get("fork")
    upstream = ok_results.get("upstream")
    if fork and upstream:
        for metric, higher_is_better in metric_specs:
            fork_value = fork["summary"].get(metric)
            upstream_value = upstream["summary"].get(metric)
            comparison_rows.append({
                "metric": metric,
                "upstream": upstream_value,
                "fork": fork_value,
                "fork_vs_upstream": ratio_text(fork_value, upstream_value, higher_is_better),
            })

    sections = [
        "# Upstream vs Fork E2E Benchmark",
        "## Config",
        markdown_rows(config_rows, ["metric", "value"]),
    ]
    for result in results:
        sections.append(f"## {result['label']} status")
        if result.get("status") == "ok":
            summary_rows = [{"metric": key, "value": value} for key, value in result["summary"].items()]
            sections.append(markdown_rows(summary_rows, ["metric", "value"]))
            run_columns = [
                "run",
                "elapsed_s",
                "ttft_s",
                "decode_tokens_per_s",
                "itl_ms_avg",
                "decode_step_ms_p95",
                "peak_gpu_memory_gb",
            ]
            sections.append("### Per-run")
            sections.append(markdown_rows(result["rows"], run_columns))
        else:
            error_text = result.get("error", "")
            stderr = result.get("stderr", "").strip()
            stdout = result.get("stdout", "").strip()
            sections.append(f"Status: failed, returncode={result.get('returncode')}")
            if error_text:
                sections.append(f"Error: {error_text}")
            if stderr:
                sections.append("```text\n" + stderr[-4000:] + "\n```")
            if stdout:
                sections.append("```text\n" + stdout[-4000:] + "\n```")

    if comparison_rows:
        sections.extend([
            "## Direct Comparison",
            markdown_rows(comparison_rows, ["metric", "upstream", "fork", "fork_vs_upstream"]),
            "",
            "For latency and memory metrics, `fork_vs_upstream` is computed as upstream / fork. "
            "For throughput metrics, it is computed as fork / upstream.",
        ])
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description="Compare this nano-vLLM fork against an upstream checkout.")
    parser.add_argument("--fork-repo", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--upstream-repo", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--num-prompts", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--output", default="upstream_vs_fork.md")
    args = parser.parse_args()

    assert args.repeat >= 1
    assert args.warmup >= 0

    fork_repo = Path(args.fork_repo)
    upstream_repo = Path(args.upstream_repo)
    if not fork_repo.exists():
        raise SystemExit(f"fork repo does not exist: {fork_repo}")
    if not upstream_repo.exists():
        raise SystemExit(f"upstream repo does not exist: {upstream_repo}")

    results = [
        run_repo("upstream", upstream_repo, args),
        run_repo("fork", fork_repo, args),
    ]
    text = build_report(results, args)
    print(text)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(f"\nSaved upstream comparison to {output_path}")


if __name__ == "__main__":
    main()
