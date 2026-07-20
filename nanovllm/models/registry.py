from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class ModelRegistration:
    module: str
    class_name: str
    architectures: tuple[str, ...]
    model_types: tuple[str, ...]

    def load_model_class(self):
        module = import_module(self.module)
        return getattr(module, self.class_name)


_MODEL_REGISTRATIONS = (
    ModelRegistration(
        module="nanovllm.models.qwen3",
        class_name="Qwen3ForCausalLM",
        architectures=("Qwen3ForCausalLM",),
        model_types=("qwen3",),
    ),
    ModelRegistration(
        module="nanovllm.models.qwen3_5",
        class_name="Qwen3_5ForCausalLM",
        architectures=("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM"),
        model_types=("qwen3_5", "qwen3_5_text"),
    ),
)


def resolve_model_registration(hf_config, hf_text_config=None) -> ModelRegistration:
    architectures = tuple(getattr(hf_config, "architectures", None) or ())
    for registration in _MODEL_REGISTRATIONS:
        if any(name in registration.architectures for name in architectures):
            return registration

    model_types = {
        getattr(hf_config, "model_type", None),
        getattr(hf_text_config, "model_type", None),
    }
    model_types.discard(None)
    for registration in _MODEL_REGISTRATIONS:
        if model_types.intersection(registration.model_types):
            return registration

    raise ValueError(
        "Unsupported model architecture: "
        f"architectures={architectures!r}, model_types={sorted(model_types)!r}"
    )


def get_model_class(hf_config, hf_text_config=None):
    return resolve_model_registration(hf_config, hf_text_config).load_model_class()


def create_model(hf_config, hf_text_config=None):
    model_class = get_model_class(hf_config, hf_text_config)
    return model_class(hf_text_config or hf_config)
