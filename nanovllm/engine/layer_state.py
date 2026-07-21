from collections import deque
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LayerStateSpec:
    layer_idx: int
    layer_type: str


@dataclass(frozen=True)
class PagedKVStateSpec(LayerStateSpec):
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype

    def bytes_per_block(self, block_size: int) -> int:
        return 2 * block_size * self.num_kv_heads * self.head_dim * self.dtype.itemsize


@dataclass(frozen=True)
class DeltaNetStateSpec(LayerStateSpec):
    conv_dim: int
    conv_width: int
    num_value_heads: int
    key_head_dim: int
    value_head_dim: int
    conv_dtype: torch.dtype
    recurrent_dtype: torch.dtype = torch.float32

    def bytes_per_sequence(self) -> int:
        conv_elements = self.conv_dim * self.conv_width
        recurrent_elements = (
            self.num_value_heads * self.key_head_dim * self.value_head_dim
        )
        return (
            conv_elements * self.conv_dtype.itemsize
            + recurrent_elements * self.recurrent_dtype.itemsize
        )


@dataclass
class LayerState:
    layer_idx: int


@dataclass
class PagedKVState(LayerState):
    k_cache: torch.Tensor
    v_cache: torch.Tensor


@dataclass
class DeltaNetState(LayerState):
    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    slot_ids: torch.Tensor | None = None


class HybridStateManager:
    """Fixed-slot request state pool for all DeltaNet layers."""

    def __init__(
        self,
        specs: list[DeltaNetStateSpec],
        capacity: int,
        device: torch.device | str,
    ):
        if capacity <= 0:
            raise ValueError(f"Hybrid state capacity must be positive, got {capacity}")
        self.specs = {spec.layer_idx: spec for spec in specs}
        self.capacity = capacity
        self.device = torch.device(device)
        self.free_slots = deque(range(capacity))
        self.seq_to_slot: dict[int, int] = {}
        self.pools: dict[int, DeltaNetState] = {}
        self._diagnostics_enabled = False
        self.reset_diagnostics()
        for spec in specs:
            self.pools[spec.layer_idx] = DeltaNetState(
                layer_idx=spec.layer_idx,
                conv_state=torch.zeros(
                    capacity,
                    spec.conv_dim,
                    spec.conv_width,
                    dtype=spec.conv_dtype,
                    device=self.device,
                ),
                recurrent_state=torch.zeros(
                    capacity,
                    spec.num_value_heads,
                    spec.key_head_dim,
                    spec.value_head_dim,
                    dtype=spec.recurrent_dtype,
                    device=self.device,
                ),
            )

    @property
    def allocated_count(self) -> int:
        return len(self.seq_to_slot)

    @property
    def free_count(self) -> int:
        return len(self.free_slots)

    def set_diagnostics(self, enabled: bool, reset: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)
        if reset:
            self.reset_diagnostics()

    def reset_diagnostics(self) -> None:
        self._diagnostics = {
            "gather_calls": 0,
            "gather_layer_ops": 0,
            "gather_bytes": 0,
            "commit_calls": 0,
            "commit_layer_ops": 0,
            "commit_bytes": 0,
            "slot_id_upload_bytes": 0,
            "batch_sizes": [],
        }

    def get_diagnostics(self) -> dict[str, object]:
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._diagnostics.items()
        }

    def allocate(self, seq_ids: list[int]) -> None:
        missing = [seq_id for seq_id in seq_ids if seq_id not in self.seq_to_slot]
        if len(missing) > len(self.free_slots):
            raise RuntimeError(
                f"DeltaNet state pool exhausted: requested={len(missing)}, "
                f"free={len(self.free_slots)}, capacity={self.capacity}"
            )
        for seq_id in missing:
            slot = self.free_slots.popleft()
            self.seq_to_slot[seq_id] = slot
            for pool in self.pools.values():
                pool.conv_state[slot].zero_()
                pool.recurrent_state[slot].zero_()

    def free(self, seq_ids: list[int]) -> None:
        for seq_id in seq_ids:
            slot = self.seq_to_slot.pop(seq_id, None)
            if slot is None:
                continue
            for pool in self.pools.values():
                pool.conv_state[slot].zero_()
                pool.recurrent_state[slot].zero_()
            self.free_slots.append(slot)

    def gather(self, seq_ids: list[int]) -> dict[int, DeltaNetState]:
        missing = [seq_id for seq_id in seq_ids if seq_id not in self.seq_to_slot]
        if missing:
            raise KeyError(f"DeltaNet state is not allocated for sequence ids {missing}")
        slot_ids = torch.tensor(
            [self.seq_to_slot[seq_id] for seq_id in seq_ids],
            dtype=torch.long,
            device=self.device,
        )
        states = {
            layer_idx: DeltaNetState(
                layer_idx=layer_idx,
                conv_state=pool.conv_state.index_select(0, slot_ids),
                recurrent_state=pool.recurrent_state.index_select(0, slot_ids),
                slot_ids=slot_ids,
            )
            for layer_idx, pool in self.pools.items()
        }
        if self._diagnostics_enabled:
            gathered_bytes = sum(
                state.conv_state.numel() * state.conv_state.element_size()
                + state.recurrent_state.numel()
                * state.recurrent_state.element_size()
                for state in states.values()
            )
            self._diagnostics["gather_calls"] += 1
            self._diagnostics["gather_layer_ops"] += len(states)
            self._diagnostics["gather_bytes"] += gathered_bytes
            self._diagnostics["slot_id_upload_bytes"] += (
                slot_ids.numel() * slot_ids.element_size()
            )
            self._diagnostics["batch_sizes"].append(len(seq_ids))
        return states

    def commit(self, states: dict[int, DeltaNetState]) -> None:
        committed_bytes = 0
        for layer_idx, state in states.items():
            if state.slot_ids is None:
                raise ValueError(f"Layer {layer_idx} state has no slot ids")
            pool = self.pools[layer_idx]
            pool.conv_state.index_copy_(0, state.slot_ids, state.conv_state)
            pool.recurrent_state.index_copy_(0, state.slot_ids, state.recurrent_state)
            if self._diagnostics_enabled:
                committed_bytes += (
                    state.conv_state.numel() * state.conv_state.element_size()
                    + state.recurrent_state.numel()
                    * state.recurrent_state.element_size()
                )
        if self._diagnostics_enabled:
            self._diagnostics["commit_calls"] += 1
            self._diagnostics["commit_layer_ops"] += len(states)
            self._diagnostics["commit_bytes"] += committed_bytes

    def reorder(self, seq_ids: list[int]) -> dict[int, DeltaNetState]:
        return self.gather(seq_ids)


def delta_state_bytes_per_sequence(specs: list[LayerStateSpec]) -> int:
    return sum(
        spec.bytes_per_sequence()
        for spec in specs
        if isinstance(spec, DeltaNetStateSpec)
    )


def paged_kv_bytes_per_block(specs: list[LayerStateSpec], block_size: int) -> int:
    return sum(
        spec.bytes_per_block(block_size)
        for spec in specs
        if isinstance(spec, PagedKVStateSpec)
    )
