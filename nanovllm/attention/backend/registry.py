from __future__ import annotations

from dataclasses import dataclass

from nanovllm.attention.backend.base import BaseAttentionBackend


@dataclass(frozen=True, slots=True)
class AttentionBackendRegistration:
    layer_type: str
    name: str
    backend_cls: type[BaseAttentionBackend]
    module_attr: str


_BACKENDS: dict[tuple[str, str], AttentionBackendRegistration] = {}
_DEFAULTS: dict[str, str] = {}


def register_attention_backend(
    *,
    layer_type: str,
    name: str,
    backend_cls: type[BaseAttentionBackend],
    module_attr: str,
    default: bool = False,
) -> None:
    key = (layer_type, name)
    registration = AttentionBackendRegistration(
        layer_type=layer_type,
        name=name,
        backend_cls=backend_cls,
        module_attr=module_attr,
    )
    existing = _BACKENDS.get(key)
    if existing is not None and existing != registration:
        raise ValueError(
            f"Attention backend {layer_type!r}/{name!r} is already registered"
        )
    _BACKENDS[key] = registration
    if default:
        previous = _DEFAULTS.get(layer_type)
        if previous is not None and previous != name:
            raise ValueError(
                f"Default backend for {layer_type!r} is already {previous!r}"
            )
        _DEFAULTS[layer_type] = name


def get_attention_backend_registration(
    layer_type: str,
    name: str | None = None,
) -> AttentionBackendRegistration:
    backend_name = name or _DEFAULTS.get(layer_type)
    if backend_name is None:
        available = sorted(key_name for key_type, key_name in _BACKENDS if key_type == layer_type)
        raise ValueError(
            f"No default backend for layer type {layer_type!r}; "
            f"available={available}"
        )
    try:
        return _BACKENDS[(layer_type, backend_name)]
    except KeyError as exc:
        available = sorted(key_name for key_type, key_name in _BACKENDS if key_type == layer_type)
        raise ValueError(
            f"Unknown backend {backend_name!r} for layer type {layer_type!r}; "
            f"available={available}"
        ) from exc


def create_attention_backend(
    layer_type: str,
    name: str | None = None,
) -> BaseAttentionBackend:
    registration = get_attention_backend_registration(layer_type, name)
    return registration.backend_cls()
