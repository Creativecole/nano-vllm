import os
from dataclasses import dataclass, field
from glob import glob
from collections.abc import Iterable

import torch
from torch import nn
from safetensors import safe_open


@dataclass
class WeightLoadReport:
    loaded: set[str] = field(default_factory=set)
    missing: set[str] = field(default_factory=set)
    duplicate: set[str] = field(default_factory=set)
    unexpected_text_weights: set[str] = field(default_factory=set)
    intentionally_skipped_non_text: set[str] = field(default_factory=set)
    tied_aliases: set[str] = field(default_factory=set)

    @property
    def ok(self) -> bool:
        return not (self.missing or self.duplicate or self.unexpected_text_weights)

    def summary(self) -> str:
        return (
            f"loaded={len(self.loaded)}, missing={len(self.missing)}, "
            f"duplicate={len(self.duplicate)}, "
            f"unexpected_text={len(self.unexpected_text_weights)}, "
            f"skipped_non_text={len(self.intentionally_skipped_non_text)}, "
            f"tied_aliases={len(self.tied_aliases)}"
        )


class WeightLoadingError(RuntimeError):
    def __init__(self, report: WeightLoadReport):
        details = [f"Strict weight loading failed: {report.summary()}"]
        for label, values in (
            ("missing", report.missing),
            ("duplicate", report.duplicate),
            ("unexpected text weights", report.unexpected_text_weights),
        ):
            if values:
                preview = ", ".join(sorted(values)[:20])
                details.append(f"{label}: {preview}")
        super().__init__("\n".join(details))
        self.report = report


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def _named_parameters_with_aliases(model: nn.Module) -> dict[str, nn.Parameter]:
    try:
        return dict(model.named_parameters(remove_duplicate=False))
    except TypeError:
        return dict(model.named_parameters())


def _is_explicitly_skipped(model: nn.Module, weight_name: str) -> bool:
    exact_names = getattr(model, "intentionally_skipped_weight_names", ())
    prefixes = getattr(model, "intentionally_skipped_weight_prefixes", ())
    return weight_name in exact_names or any(weight_name.startswith(prefix) for prefix in prefixes)


def _map_text_prefix(model: nn.Module, weight_name: str) -> str:
    for checkpoint_prefix, model_prefix in getattr(model, "checkpoint_prefix_mapping", ()):
        if weight_name.startswith(checkpoint_prefix):
            return model_prefix + weight_name[len(checkpoint_prefix):]
    return weight_name


def _replace_module_component(name: str, source: str, target: str) -> str | None:
    components = name.split(".")
    try:
        index = components.index(source)
    except ValueError:
        return None
    components[index] = target
    return ".".join(components)


def _map_packed_parameter(model: nn.Module, weight_name: str):
    for source, (target, shard_id) in getattr(model, "packed_modules_mapping", {}).items():
        mapped_name = _replace_module_component(weight_name, source, target)
        if mapped_name is not None:
            return mapped_name, shard_id
    return weight_name, None


def _parameter_aliases(parameters: dict[str, nn.Parameter]) -> dict[int, set[str]]:
    aliases: dict[int, set[str]] = {}
    for name, parameter in parameters.items():
        aliases.setdefault(id(parameter), set()).add(name)
    return aliases


def _canonical_parameter_name(names: set[str]) -> str:
    if "model.embed_tokens.weight" in names:
        return "model.embed_tokens.weight"
    return sorted(names)[0]


def _expected_target_keys(
    model: nn.Module,
    parameters: dict[str, nn.Parameter],
) -> tuple[set[tuple[int, object]], dict[tuple[int, object], str]]:
    packed_mapping = getattr(model, "packed_modules_mapping", {})
    expected: set[tuple[int, object]] = set()
    labels: dict[tuple[int, object], str] = {}
    for parameter_name, parameter in parameters.items():
        components = set(parameter_name.split("."))
        packed_shards = {
            shard_id
            for target, shard_id in packed_mapping.values()
            if target in components
        }
        if not packed_shards:
            packed_shards = {None}
        for shard_id in packed_shards:
            key = (id(parameter), shard_id)
            expected.add(key)
            suffix = "" if shard_id is None else f"[{shard_id}]"
            labels.setdefault(key, f"{parameter_name}{suffix}")
    return expected, labels


def load_weights(
    model: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    strict: bool = True,
) -> WeightLoadReport:
    """Load checkpoint tensors with an audited text-only mapping."""
    parameters = _named_parameters_with_aliases(model)
    aliases_by_id = _parameter_aliases(parameters)
    expected_targets, expected_target_labels = _expected_target_keys(model, parameters)
    report = WeightLoadReport()
    seen_targets: set[tuple[int, object]] = set()

    for checkpoint_name, loaded_weight in weights:
        if _is_explicitly_skipped(model, checkpoint_name):
            report.intentionally_skipped_non_text.add(checkpoint_name)
            continue

        mapped_name = _map_text_prefix(model, checkpoint_name)
        mapped_name, shard_id = _map_packed_parameter(model, mapped_name)
        parameter = parameters.get(mapped_name)
        if parameter is None:
            report.unexpected_text_weights.add(checkpoint_name)
            continue

        parameter_id = id(parameter)
        target_key = (parameter_id, shard_id)
        if target_key in seen_targets:
            aliases = aliases_by_id.get(parameter_id, set())
            is_tied_alias = (
                bool(getattr(model, "tie_word_embeddings", False))
                and shard_id is None
                and mapped_name in aliases
                and len(aliases) > 1
                and mapped_name not in report.loaded
            )
            if is_tied_alias:
                report.tied_aliases.add(checkpoint_name)
                report.loaded.add(mapped_name)
                continue
            report.duplicate.add(checkpoint_name)
            continue

        weight_loader = getattr(parameter, "weight_loader", default_weight_loader)
        if shard_id is None:
            weight_loader(parameter, loaded_weight)
        else:
            weight_loader(parameter, loaded_weight, shard_id)
        seen_targets.add(target_key)
        report.loaded.add(mapped_name)

    for target_key in expected_targets - seen_targets:
        parameter_id, shard_id = target_key
        aliases = aliases_by_id[parameter_id]
        if shard_id is None:
            report.missing.add(_canonical_parameter_name(aliases))
        else:
            report.missing.add(expected_target_labels[target_key])

    if strict and not report.ok:
        raise WeightLoadingError(report)
    return report


def _iter_safetensor_weights(path: str):
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files found in {path}")
    for file in files:
        with safe_open(file, "pt", "cpu") as handle:
            for weight_name in handle.keys():
                yield weight_name, handle.get_tensor(weight_name)


def load_model(model: nn.Module, path: str, *, strict: bool = True) -> WeightLoadReport:
    report = load_weights(model, _iter_safetensor_weights(path), strict=strict)
    print(f"Weight loading: {report.summary()}", flush=True)
    return report
