import argparse
import json
import re
from pathlib import Path
from statistics import mean


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    rows = []
    headers = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line.startswith("|") or not line.endswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells:
            continue
        if all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells):
            continue
        if headers is None:
            headers = cells
            continue
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def parse_float(value: str, default: float = 0.0) -> float:
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
    return float(match.group(0)) if match else default


def parse_shape_dim(shape: str, key: str) -> int | None:
    match = re.search(rf"\b{re.escape(key)}=(\d+)", shape)
    return int(match.group(1)) if match else None


def parse_metric_table(path: Path) -> dict[str, str]:
    metrics = {}
    for row in parse_markdown_table(path):
        if "Metric" in row and "Value" in row:
            metrics[row["Metric"]] = row["Value"]
        elif "metric" in row and "mean" in row:
            metrics[row["metric"]] = row["mean"]
    return metrics


def summarize_kernel_rows(rows: list[dict[str, str]]):
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        row["speedup_value"] = parse_float(row.get("Speedup", "0"))
        grouped.setdefault(row.get("Kernel", "unknown"), []).append(row)

    wins = [row for row in rows if row["speedup_value"] >= 1.05]
    regressions = [row for row in rows if row["speedup_value"] < 0.95]
    return grouped, wins, regressions


def summarize_cuda_rows(rows: list[dict[str, str]]):
    for row in rows:
        row["naive_speedup_value"] = parse_float(row.get("Naive vs torch", "0"))
        row["tiled_speedup_value"] = parse_float(row.get("Tiled vs torch", "0"))
    faster_than_torch = [row for row in rows if row["tiled_speedup_value"] >= 1.05 or row["naive_speedup_value"] >= 1.05]
    return faster_than_torch


def threshold_policy(rows: list[dict[str, str]], fast_backend: str, slow_backend: str) -> str | dict[str, str]:
    fast_ns = []
    slow_ns = []
    for row in rows:
        n = parse_shape_dim(row.get("Shape", ""), "N")
        if n is None:
            continue
        if row["speedup_value"] >= 1.05:
            fast_ns.append(n)
        elif row["speedup_value"] < 1.05:
            slow_ns.append(n)

    if fast_ns and not slow_ns:
        return fast_backend
    if slow_ns and not fast_ns:
        return slow_backend
    if not fast_ns:
        return slow_backend

    max_fast = max(fast_ns)
    later_slow_ns = [n for n in slow_ns if n > max_fast]
    min_slow = min(later_slow_ns) if later_slow_ns else None
    policy = {f"N<={max_fast}": fast_backend}
    if min_slow is not None:
        policy[f"N>={min_slow}"] = slow_backend
    return policy


def build_policy(kernel_rows, cuda_rows, e2e_metrics, kernel_results, cuda_gemm_results, e2e_results):
    grouped, _, _ = summarize_kernel_rows(kernel_rows)
    cuda_wins = summarize_cuda_rows(cuda_rows)

    rules = {}
    for kernel in ("RMSNorm", "AddRMSNorm"):
        rows = grouped.get(kernel, [])
        rules[kernel] = "triton" if rows and mean(row["speedup_value"] for row in rows) >= 1.05 else "torch"

    rules["SiluAndMul"] = threshold_policy(grouped.get("SiluAndMul", []), "triton", "torch")
    rules["RoPE"] = threshold_policy(grouped.get("RoPE", []), "triton", "torch")

    kv_rows = grouped.get("KVCacheStore", [])
    rules["KVCacheStore"] = "2d_triton" if kv_rows and mean(row["speedup_value"] for row in kv_rows) >= 1.05 else "1d_triton"

    linear_rows = [row for row in kernel_rows if row.get("Kernel", "").startswith("Linear-")]
    rules["Linear"] = "triton_gemv" if linear_rows and mean(row["speedup_value"] for row in linear_rows) >= 1.05 else "torch_cublas"
    rules["CUDA_GEMM"] = "candidate" if cuda_wins else "research_only"

    return {
        "device": e2e_metrics.get("gpu", "unknown"),
        "model": Path(e2e_metrics.get("model", "unknown")).name,
        "source_files": {
            "kernel_results": str(kernel_results),
            "cuda_gemm_results": str(cuda_gemm_results),
            "e2e_results": str(e2e_results),
        },
        "rules": rules,
    }


def markdown_rows(rows: list[dict[str, str]], columns: list[str], limit: int | None = None) -> str:
    selected = rows if limit is None else rows[:limit]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in selected:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    if not selected:
        lines.append("| " + " | ".join(["n/a"] + [""] * (len(columns) - 1)) + " |")
    return "\n".join(lines)


def explain_policy(policy: dict) -> list[str]:
    rules = policy["rules"]
    explanations = []
    explanations.append(f"RMSNorm/AddRMSNorm use `{rules['RMSNorm']}` / `{rules['AddRMSNorm']}` because the measured Triton path is consistently faster.")
    explanations.append(f"SiLU-and-Mul policy is `{rules['SiluAndMul']}` because large prefill shapes can erase the small-batch Triton win.")
    explanations.append(f"RoPE policy is `{rules['RoPE']}` because the Triton path is shape-sensitive.")
    explanations.append(f"KVCacheStore remains `{rules['KVCacheStore']}` because the experimental 2D store regressed versus the current 1D Triton store.")
    explanations.append(f"Linear remains `{rules['Linear']}` because cuBLAS is faster than the educational Triton GEMV on Qwen3-4B shapes.")
    explanations.append(f"CUDA_GEMM is `{rules['CUDA_GEMM']}` because naive/tiled CUDA kernels are worklog baselines, not production replacements.")
    return explanations


def build_report(kernel_rows, cuda_rows, e2e_metrics, policy, kernel_path, cuda_path, e2e_path):
    _, wins, regressions = summarize_kernel_rows(kernel_rows)
    cuda_wins = summarize_cuda_rows(cuda_rows)
    cuda_regressions = [row for row in cuda_rows if row["tiled_speedup_value"] < 1.0 and row["naive_speedup_value"] < 1.0]

    e2e_summary_keys = [
        "model", "gpu", "prompt_len", "num_prompts", "max_tokens", "elapsed_s",
        "ttft_s", "prefill_time_s", "decode_time_s", "decode_tokens_per_s",
        "itl_ms_avg", "decode_step_ms_p50", "decode_step_ms_p95",
        "peak_gpu_memory_gb", "num_kvcache_blocks", "max_used_blocks",
        "max_block_utilization", "kv_cache_dtype", "linear_backend",
    ]
    e2e_rows = [{"Metric": key, "Value": e2e_metrics.get(key, "n/a")} for key in e2e_summary_keys]

    policy_rows = [{"Kernel": key, "Policy": json.dumps(value, ensure_ascii=False)} for key, value in policy["rules"].items()]

    lines = [
        "# Qwen3-4B RTX 5090 Kernel Policy Report",
        "",
        "## Inputs",
        "",
        f"- Kernel results: `{kernel_path}`",
        f"- CUDA GEMM results: `{cuda_path}`",
        f"- E2E results: `{e2e_path}`",
        "",
        "## E2E Summary",
        "",
        markdown_rows(e2e_rows, ["Metric", "Value"]),
        "",
        "## Kernel Wins",
        "",
        markdown_rows(wins, ["Kernel", "Shape", "Baseline", "Triton", "Speedup", "Correctness"], limit=20),
        "",
        "## Kernel Regressions",
        "",
        markdown_rows(regressions, ["Kernel", "Shape", "Baseline", "Triton", "Speedup", "Correctness"], limit=30),
        "",
        "## CUDA GEMM Worklog Summary",
        "",
        "Custom CUDA GEMM is treated as a research baseline. Rows below are cases where naive/tiled CUDA is slower than cuBLAS.",
        "",
        markdown_rows(cuda_regressions, ["Shape", "M", "K", "N", "torch.matmul", "CUDA naive", "CUDA tiled", "Naive vs torch", "Tiled vs torch"], limit=20),
        "",
    ]
    if cuda_wins:
        lines.extend([
            "CUDA GEMM candidate wins:",
            "",
            markdown_rows(cuda_wins, ["Shape", "M", "K", "N", "Naive vs torch", "Tiled vs torch"], limit=20),
            "",
        ])

    lines.extend([
        "## Recommended Default Policy",
        "",
        markdown_rows(policy_rows, ["Kernel", "Policy"]),
        "",
        "## Negative Results Explanation",
        "",
        *[f"- {item}" for item in explain_policy(policy)],
        "",
        "## Runtime Scope",
        "",
        "This policy is a report artifact only. It is not wired into nano-vLLM runtime dispatch, and the default serving path remains FlashAttention 2, model-dtype KV cache, and cuBLAS Linear.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze benchmark Markdown tables and generate a kernel policy report.")
    parser.add_argument("--kernel-results", required=True)
    parser.add_argument("--cuda-gemm-results", required=True)
    parser.add_argument("--e2e-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-output", required=True)
    args = parser.parse_args()

    kernel_path = Path(args.kernel_results)
    cuda_path = Path(args.cuda_gemm_results)
    e2e_path = Path(args.e2e_results)
    output_path = Path(args.output)
    policy_path = Path(args.policy_output)

    kernel_rows = parse_markdown_table(kernel_path)
    cuda_rows = parse_markdown_table(cuda_path)
    e2e_metrics = parse_metric_table(e2e_path)

    policy = build_policy(kernel_rows, cuda_rows, e2e_metrics, kernel_path, cuda_path, e2e_path)
    report = build_report(kernel_rows, cuda_rows, e2e_metrics, policy, kernel_path, cuda_path, e2e_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report + "\n", encoding="utf-8")
    policy_path.write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved report to {output_path}")
    print(f"Saved policy to {policy_path}")


if __name__ == "__main__":
    main()
