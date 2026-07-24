from __future__ import annotations

import torch

from nanovllm.attention.backend.base import BaseAttentionBackend
from nanovllm.attention.metadata import FullAttentionMetadata
from nanovllm.engine.layer_state import PagedKVState, PagedKVStateSpec


class FullAttentionBackend(BaseAttentionBackend):
    layer_type = "full_attention"
    profile_range_name = "qwen35_full_attention_mixer"

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
        if metadata is not None and not isinstance(metadata, FullAttentionMetadata):
            raise TypeError(
                "FullAttentionBackend requires FullAttentionMetadata, got "
                f"{type(metadata).__name__}"
            )
        return mixer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            runtime_metadata=metadata,
        )

    def get_state_spec(self, mixer, dtype: torch.dtype) -> PagedKVStateSpec:
        return PagedKVStateSpec(
            layer_idx=mixer.layer_idx,
            layer_type=self.layer_type,
            num_kv_heads=mixer.num_kv_heads,
            head_dim=mixer.head_dim,
            dtype=dtype,
        )

    def enable_runtime(self, mixer) -> None:
        mixer.enable_paged_attention()

    def bind_state(self, mixer, state: PagedKVState) -> None:
        if not isinstance(state, PagedKVState):
            raise TypeError(
                f"Full attention requires PagedKVState, got {type(state).__name__}"
            )
        mixer.bind_paged_state(state)
