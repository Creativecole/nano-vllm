from __future__ import annotations

import torch

from nanovllm.attention.backend.base import BaseAttentionBackend
from nanovllm.attention.metadata import DeltaNetMetadata
from nanovllm.engine.layer_state import DeltaNetStateSpec


class DeltaNetBackend(BaseAttentionBackend):
    layer_type = "linear_attention"
    profile_range_name = "qwen35_deltanet_mixer"

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
        if metadata is not None and not isinstance(metadata, DeltaNetMetadata):
            raise TypeError(
                "DeltaNetBackend requires DeltaNetMetadata, got "
                f"{type(metadata).__name__}"
            )
        return mixer(
            hidden_states,
            attention_mask=attention_mask,
            layer_state=layer_state,
            runtime_metadata=metadata,
        )

    def get_state_spec(self, mixer, dtype: torch.dtype) -> DeltaNetStateSpec:
        return DeltaNetStateSpec(
            layer_idx=mixer.layer_idx,
            layer_type=self.layer_type,
            conv_dim=mixer.conv_dim,
            conv_width=mixer.conv_kernel_size,
            num_value_heads=mixer.num_v_heads,
            key_head_dim=mixer.head_k_dim,
            value_head_dim=mixer.head_v_dim,
            conv_dtype=dtype,
        )
