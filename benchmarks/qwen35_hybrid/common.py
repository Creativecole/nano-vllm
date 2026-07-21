from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from statistics import mean, median
from typing import Iterable

import torch
from transformers import AutoConfig

from nanovllm.config import get_hf_text_config, resolve_torch_dtype


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmarks/qwen35_hybrid/results"


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"Expected comma-separated positive integers, got {value!r}")
    return values


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(values: Iterable[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"mean": None, "p50": None, "p95": None}
    return {
        "mean": mean(clean),
        "p50": median(clean),
        "p95": percentile(clean, 0.95),
    }


def write_json(path: str | Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, text.rstrip() + "\n")


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    def render(value):
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(render(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(model: str) -> dict[str, object]:
    return {
        "model": model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": _package_version("transformers"),
        "vllm": _package_version("vllm"),
        "git_commit": git_commit(),
    }


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def load_model_facts(model: str) -> dict[str, object]:
    outer = AutoConfig.from_pretrained(model)
    text = get_hf_text_config(outer)
    dtype_value = getattr(text, "dtype", None) or getattr(text, "torch_dtype", None)
    dtype = resolve_torch_dtype(dtype_value)
    layer_types = list(text.layer_types)
    return {
        "outer_config_class": type(outer).__name__,
        "text_config_class": type(text).__name__,
        "model_type": text.model_type,
        "vocab_size": text.vocab_size,
        "hidden_size": text.hidden_size,
        "num_hidden_layers": text.num_hidden_layers,
        "num_attention_heads": text.num_attention_heads,
        "num_key_value_heads": text.num_key_value_heads,
        "head_dim": text.head_dim,
        "intermediate_size": text.intermediate_size,
        "dtype": str(dtype),
        "layer_types": layer_types,
        "full_attention_layers": layer_types.count("full_attention"),
        "linear_attention_layers": layer_types.count("linear_attention"),
        "linear_num_key_heads": text.linear_num_key_heads,
        "linear_num_value_heads": text.linear_num_value_heads,
        "linear_key_head_dim": text.linear_key_head_dim,
        "linear_value_head_dim": text.linear_value_head_dim,
        "linear_conv_kernel_dim": text.linear_conv_kernel_dim,
    }


def load_text_configs(model: str):
    outer = AutoConfig.from_pretrained(model)
    text = get_hf_text_config(outer)
    text._attn_implementation = "eager"
    return outer, text


def _construct_on_device(model_class, config, device: str):
    previous_dtype = torch.get_default_dtype()
    previous_device = torch.get_default_device()
    dtype_value = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    dtype = resolve_torch_dtype(dtype_value)
    try:
        torch.set_default_dtype(dtype)
        torch.set_default_device(device)
        return model_class(config).eval()
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)


def load_hf_text_reference(model: str, device: str = "cuda"):
    from transformers.models.qwen3_5 import modeling_qwen3_5

    from nanovllm.utils.loader import load_model

    _, text_config = load_text_configs(model)
    reference = _construct_on_device(
        modeling_qwen3_5.Qwen3_5ForCausalLM, text_config, device
    )
    reference.checkpoint_prefix_mapping = (
        ("model.language_model.", "model."),
        ("language_model.", "model."),
    )
    reference.intentionally_skipped_weight_prefixes = (
        "model.visual.",
        "visual.",
        "model.multi_modal_projector.",
        "multi_modal_projector.",
        "mtp.",
        "model.mtp.",
        "draft_model.",
        "auxiliary_head.",
    )
    reference.tie_word_embeddings = bool(text_config.tie_word_embeddings)
    report = load_model(reference, model, strict=True)
    for layer in reference.model.layers:
        mixer = getattr(layer, "linear_attn", None)
        if mixer is None:
            continue
        mixer.causal_conv1d_fn = None
        mixer.causal_conv1d_update = modeling_qwen3_5.torch_causal_conv1d_update
        mixer.chunk_gated_delta_rule = modeling_qwen3_5.torch_chunk_gated_delta_rule
        mixer.recurrent_gated_delta_rule = (
            modeling_qwen3_5.torch_recurrent_gated_delta_rule
        )
    return reference, text_config, report


def load_nano_text_reference(model: str, device: str = "cuda"):
    from nanovllm.models.qwen3_5 import Qwen3_5ForCausalLM
    from nanovllm.utils.loader import load_model

    _, text_config = load_text_configs(model)
    reference = _construct_on_device(Qwen3_5ForCausalLM, text_config, device)
    report = load_model(reference, model, strict=True)
    return reference, text_config, report


def deterministic_prompts(
    vocab_size: int,
    batch_size: int,
    prompt_len: int,
    seed: int = 17,
) -> list[list[int]]:
    usable_vocab = max(2, vocab_size - 1)
    return [
        [1 + ((seed + row * 97 + position * 31) % usable_vocab) for position in range(prompt_len)]
        for row in range(batch_size)
    ]


def theoretical_cache_bytes(
    facts: dict[str, object],
    batch_size: int,
    total_sequence_length: int,
) -> tuple[int, int]:
    dtype = resolve_torch_dtype(str(facts["dtype"]))
    kv_bytes = (
        int(facts["full_attention_layers"])
        * 2
        * batch_size
        * total_sequence_length
        * int(facts["num_key_value_heads"])
        * int(facts["head_dim"])
        * dtype.itemsize
    )
    conv_dim = (
        int(facts["linear_num_key_heads"])
        * int(facts["linear_key_head_dim"])
        * 2
        + int(facts["linear_num_value_heads"])
        * int(facts["linear_value_head_dim"])
    )
    per_layer_delta = (
        conv_dim * int(facts["linear_conv_kernel_dim"]) * dtype.itemsize
        + int(facts["linear_num_value_heads"])
        * int(facts["linear_key_head_dim"])
        * int(facts["linear_value_head_dim"])
        * torch.float32.itemsize
    )
    delta_bytes = (
        int(facts["linear_attention_layers"]) * batch_size * per_layer_delta
    )
    return kv_bytes, delta_bytes


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    relative = difference / expected.abs().clamp_min(1e-6)
    return {
        "max_abs_error": difference.max().item(),
        "mean_abs_error": difference.mean().item(),
        "max_relative_error": relative.max().item(),
        "mean_relative_error": relative.mean().item(),
    }


def format_bytes(value: int | None) -> str:
    if value is None:
        return "N/A"
    return f"{value / 2**30:.3f} GiB"


def safe_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}
