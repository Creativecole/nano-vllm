from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseAttentionBackend(ABC):
    """Execution adapter for one hybrid sequence-mixer type.

    Request allocation is intentionally not part of this interface. Cache/state
    lifetime belongs to the runtime coordinator, while the backend owns typed
    metadata validation and mixer execution.
    """

    layer_type: str
    profile_range_name: str

    @abstractmethod
    def forward(
        self,
        mixer,
        hidden_states: torch.Tensor,
        *,
        position_embeddings,
        attention_mask,
        layer_state,
        metadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_state_spec(self, mixer, dtype: torch.dtype):
        raise NotImplementedError

    def enable_runtime(self, mixer) -> None:
        return None

    def bind_state(self, mixer, state: Any) -> None:
        return None
