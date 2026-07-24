from __future__ import annotations

import torch

from nanovllm.engine.layer_state import (
    DeltaNetState,
    DeltaNetStateSpec,
    HybridStateManager,
    LayerStateSpec,
    PagedKVState,
    PagedKVStateSpec,
)
from nanovllm.utils.profiler import profile_range


class HybridCacheCoordinator:
    """Owns physical state pools for hybrid sequence mixers.

    The scheduler continues to account token capacity through the existing
    BlockManager. This coordinator owns model-execution state: shared paged KV
    tensors for full-attention layers and request-scoped DeltaNet state slots.
    """

    def __init__(
        self,
        specs: list[LayerStateSpec],
        *,
        state_capacity: int,
        num_kv_blocks: int,
        block_size: int,
        device: torch.device | str,
        compact_delta_slots: bool,
    ):
        self.specs = tuple(specs)
        self.device = torch.device(device)
        self.block_size = block_size
        self.num_kv_blocks = num_kv_blocks
        delta_specs = [
            spec for spec in specs if isinstance(spec, DeltaNetStateSpec)
        ]
        paged_specs = [
            spec for spec in specs if isinstance(spec, PagedKVStateSpec)
        ]
        if not delta_specs or not paged_specs:
            raise ValueError(
                "Hybrid cache coordination requires DeltaNet and paged-KV specs"
            )

        self.delta_states = HybridStateManager(
            delta_specs,
            capacity=state_capacity,
            device=self.device,
            compact_slots=compact_delta_slots,
        )
        self.paged_kv_states = {
            spec.layer_idx: PagedKVState(
                layer_idx=spec.layer_idx,
                k_cache=torch.empty(
                    num_kv_blocks,
                    block_size,
                    spec.num_kv_heads,
                    spec.head_dim,
                    dtype=spec.dtype,
                    device=self.device,
                ),
                v_cache=torch.empty(
                    num_kv_blocks,
                    block_size,
                    spec.num_kv_heads,
                    spec.head_dim,
                    dtype=spec.dtype,
                    device=self.device,
                ),
            )
            for spec in paged_specs
        }

    @property
    def state_capacity(self) -> int:
        return self.delta_states.capacity

    @property
    def state_layout_version(self) -> int:
        return self.delta_states.mapping_version

    def bind_model(self, model) -> None:
        model.enable_paged_attention()
        model.bind_paged_kv_states(self.paged_kv_states)

    def allocate_requests(self, request_ids: list[int]) -> None:
        self.delta_states.allocate(request_ids)

    def free_requests(self, request_ids: list[int]) -> None:
        self.delta_states.free(request_ids)

    def prepare_request_batch(
        self,
        request_ids: list[int],
        *,
        allow_reorder: bool,
    ) -> tuple[list[int], dict[int, DeltaNetState], bool]:
        self.allocate_requests(request_ids)
        if not self.delta_states.compact_slots:
            with profile_range("qwen35_state_gather"):
                states = self.delta_states.gather(request_ids)
            return request_ids, states, False

        resident_ids = self.delta_states.resident_order(request_ids)
        if resident_ids is not None and (
            allow_reorder or resident_ids == request_ids
        ):
            with profile_range("qwen35_state_resident_view"):
                states = self.delta_states.resident(resident_ids)
            return resident_ids, states, True

        with profile_range("qwen35_state_gather"):
            states = self.delta_states.gather(request_ids)
        return request_ids, states, False

    def finish_request_batch(
        self,
        states: dict[int, DeltaNetState],
        *,
        is_resident: bool,
    ) -> bool:
        if is_resident:
            self.delta_states.finish_resident(states)
            return False
        with profile_range("qwen35_state_commit"):
            self.delta_states.commit(states)
        return True

    def get_stats(self) -> dict[str, int]:
        return {
            "state_capacity": self.state_capacity,
            "allocated_state_slots": self.delta_states.allocated_count,
            "free_state_slots": self.delta_states.free_count,
            "num_kv_blocks": self.num_kv_blocks,
            "full_attention_layers": len(self.paged_kv_states),
            "deltanet_layers": len(self.delta_states.specs),
        }
