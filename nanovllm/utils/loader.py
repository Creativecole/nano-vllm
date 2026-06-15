import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def normalize_weight_name(weight_name: str) -> str:
    prefix_rewrites = (
        ("model.language_model.", "model."),
        ("language_model.", ""),
        ("base_model.", ""),
    )
    for prefix, replacement in prefix_rewrites:
        if weight_name.startswith(prefix):
            return replacement + weight_name[len(prefix):]
    return weight_name


def get_parameter_or_raise(model: nn.Module, param_name: str, original_name: str) -> nn.Parameter:
    try:
        return model.get_parameter(param_name)
    except AttributeError as exc:
        if ".linear_attn." in param_name:
            raise RuntimeError(
                "This checkpoint contains linear-attention weights "
                f"({original_name!r}), but nano-vLLM's current Qwen3 adapter "
                "only implements dense self-attention decoder layers. Use a "
                "dense Qwen3/Qwen3.5 checkpoint, or add a dedicated model "
                "adapter that implements the checkpoint's linear_attn block."
            ) from exc
        raise AttributeError(
            f"Checkpoint weight {original_name!r} was normalized to {param_name!r}, "
            "but the current model has no matching parameter. This usually means "
            "the checkpoint architecture does not match the nano-vLLM model adapter."
        ) from exc


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                param_name = normalize_weight_name(weight_name)
                for k in packed_modules_mapping:
                    if k in param_name:
                        v, shard_id = packed_modules_mapping[k]
                        packed_param_name = param_name.replace(k, v)
                        param = get_parameter_or_raise(model, packed_param_name, weight_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = get_parameter_or_raise(model, param_name, weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
