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
    is_resident: bool = False


class HybridStateManager:
    """Fixed-slot request state pool for all DeltaNet layers."""

    def __init__(
        self,
        specs: list[DeltaNetStateSpec],
        capacity: int,
        device: torch.device | str,
        compact_slots: bool = True,
    ):
        if capacity <= 0:
            raise ValueError(f"Hybrid state capacity must be positive, got {capacity}")
        self.specs = {spec.layer_idx: spec for spec in specs}
        self.capacity = capacity
        self.device = torch.device(device)
        self.compact_slots = compact_slots
        self.free_slots = deque(range(capacity)) if not compact_slots else None
        self.seq_to_slot: dict[int, int] = {}
        self.slot_to_seq: list[int | None] = [None] * capacity
        self.mapping_version = 0
        self.max_allocated_count = 0
        self.state_bytes_per_sequence = sum(
            spec.bytes_per_sequence() for spec in specs
        )
        self.pools: dict[int, DeltaNetState] = {}
        self._diagnostics_enabled = False
        self.reset_diagnostics()
        self.reset_runtime_stats()
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
        if self.free_slots is not None:
            return len(self.free_slots)
        return self.capacity - self.allocated_count

    def reset_runtime_stats(self) -> None:
        self._runtime_stats = {
            "allocation_calls": 0,
            "allocation_rows": 0,
            "allocation_zero_ops": 0,
            "resident_view_calls": 0,
            "resident_layer_views": 0,
            "resident_rows": 0,
            "estimated_state_copy_bytes_avoided": 0,
            "fallback_gather_calls": 0,
            "fallback_commit_calls": 0,
            "resident_commit_skips": 0,
            "compaction_calls": 0,
            "compaction_layer_ops": 0,
            "compaction_bytes": 0,
        }

    def get_runtime_stats(self) -> dict[str, int]:
        return dict(self._runtime_stats)

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
            "resident_view_calls": 0,
            "resident_layer_views": 0,
            "resident_rows": 0,
            "estimated_state_copy_bytes_avoided": 0,
            "resident_commit_skips": 0,
            "compaction_calls": 0,
            "compaction_layer_ops": 0,
            "compaction_bytes": 0,
        }

    def get_diagnostics(self) -> dict[str, object]:
        return {
            key: list(value) if isinstance(value, list) else value
            for key, value in self._diagnostics.items()
        }

    def allocate(self, seq_ids: list[int]) -> None:
        if len(set(seq_ids)) != len(seq_ids):
            raise ValueError("DeltaNet state request ids must be unique")
        missing = [seq_id for seq_id in seq_ids if seq_id not in self.seq_to_slot]
        if len(missing) > self.free_count:
            raise RuntimeError(
                f"DeltaNet state pool exhausted: requested={len(missing)}, "
                f"free={self.free_count}, capacity={self.capacity}"
            )
        for seq_id in missing:
            slot = (
                self.allocated_count
                if self.free_slots is None
                else self.free_slots.popleft()
            )
            if self.slot_to_seq[slot] is not None:
                raise RuntimeError(
                    f"DeltaNet resident slot {slot} is unexpectedly occupied"
            )
            self.seq_to_slot[seq_id] = slot
            self.slot_to_seq[slot] = seq_id
            if self.free_slots is not None:
                for pool in self.pools.values():
                    pool.conv_state[slot].zero_()
                    pool.recurrent_state[slot].zero_()
        if missing and self.free_slots is None:
            start = self.allocated_count - len(missing)
            stop = self.allocated_count
            for pool in self.pools.values():
                pool.conv_state[start:stop].zero_()
                pool.recurrent_state[start:stop].zero_()
        if missing:
            self.mapping_version += 1
            self._runtime_stats["allocation_calls"] += 1
            self._runtime_stats["allocation_rows"] += len(missing)
            zero_ops_per_layer = 2 if self.free_slots is None else 2 * len(missing)
            self._runtime_stats["allocation_zero_ops"] += (
                zero_ops_per_layer * len(self.pools)
            )
        self.max_allocated_count = max(
            self.max_allocated_count,
            self.allocated_count,
        )

    def free(self, seq_ids: list[int]) -> None:
        mapping_changed = False
        for seq_id in seq_ids:
            slot = self.seq_to_slot.pop(seq_id, None)
            if slot is None:
                continue
            mapping_changed = True
            if self.free_slots is not None:
                self.slot_to_seq[slot] = None
                self.free_slots.append(slot)
                continue
            last_slot = self.allocated_count
            if last_slot < 0:
                raise RuntimeError("DeltaNet state allocation count became negative")
            moved_seq_id = self.slot_to_seq[last_slot]
            if moved_seq_id is None:
                raise RuntimeError(
                    f"DeltaNet resident tail slot {last_slot} is unexpectedly empty"
                )
            if slot != last_slot:
                for pool in self.pools.values():
                    pool.conv_state[slot].copy_(pool.conv_state[last_slot])
                    pool.recurrent_state[slot].copy_(
                        pool.recurrent_state[last_slot]
                    )
                self.seq_to_slot[moved_seq_id] = slot
                self.slot_to_seq[slot] = moved_seq_id
                copied_bytes = self.state_bytes_per_sequence
                self._runtime_stats["compaction_calls"] += 1
                self._runtime_stats["compaction_layer_ops"] += len(self.pools)
                self._runtime_stats["compaction_bytes"] += copied_bytes
                if self._diagnostics_enabled:
                    self._diagnostics["compaction_calls"] += 1
                    self._diagnostics["compaction_layer_ops"] += len(self.pools)
                    self._diagnostics["compaction_bytes"] += copied_bytes
            self.slot_to_seq[last_slot] = None
        if mapping_changed:
            self.mapping_version += 1

    def resident_order(self, seq_ids: list[int]) -> list[int] | None:
        """Return a slot-ordered request list when it forms one contiguous slice."""
        if not seq_ids:
            return []
        if len(set(seq_ids)) != len(seq_ids):
            raise ValueError("DeltaNet state request ids must be unique")
        missing = [seq_id for seq_id in seq_ids if seq_id not in self.seq_to_slot]
        if missing:
            raise KeyError(f"DeltaNet state is not allocated for sequence ids {missing}")
        ordered = sorted(seq_ids, key=self.seq_to_slot.__getitem__)
        slots = [self.seq_to_slot[seq_id] for seq_id in ordered]
        if slots != list(range(slots[0], slots[0] + len(slots))):
            return None
        return ordered

    def resident(self, seq_ids: list[int]) -> dict[int, DeltaNetState]:
        """Return zero-copy state views for a contiguous, slot-ordered batch."""
        if not seq_ids:
            return {}
        slots = [self.seq_to_slot[seq_id] for seq_id in seq_ids]
        expected = list(range(slots[0], slots[0] + len(slots)))
        if slots != expected:
            raise ValueError(
                "Resident state views require contiguous slot-ordered request ids"
            )
        start, stop = slots[0], slots[-1] + 1
        states = {
            layer_idx: DeltaNetState(
                layer_idx=layer_idx,
                conv_state=pool.conv_state[start:stop],
                recurrent_state=pool.recurrent_state[start:stop],
                is_resident=True,
            )
            for layer_idx, pool in self.pools.items()
        }
        self._runtime_stats["resident_view_calls"] += 1
        self._runtime_stats["resident_layer_views"] += len(states)
        self._runtime_stats["resident_rows"] += len(seq_ids)
        avoided_bytes = 2 * len(seq_ids) * self.state_bytes_per_sequence
        self._runtime_stats["estimated_state_copy_bytes_avoided"] += avoided_bytes
        if self._diagnostics_enabled:
            self._diagnostics["resident_view_calls"] += 1
            self._diagnostics["resident_layer_views"] += len(states)
            self._diagnostics["resident_rows"] += len(seq_ids)
            self._diagnostics[
                "estimated_state_copy_bytes_avoided"
            ] += avoided_bytes
            self._diagnostics["batch_sizes"].append(len(seq_ids))
        return states

    def slot_for(self, seq_id: int) -> int:
        try:
            return self.seq_to_slot[seq_id]
        except KeyError as exc:
            raise KeyError(
                f"DeltaNet state is not allocated for sequence id {seq_id}"
            ) from exc

    def _record_fallback_gather(self) -> None:
        self._runtime_stats["fallback_gather_calls"] += 1

    def _record_fallback_commit(self) -> None:
        self._runtime_stats["fallback_commit_calls"] += 1

    def _record_resident_commit_skip(self) -> None:
        self._runtime_stats["resident_commit_skips"] += 1
        if self._diagnostics_enabled:
            self._diagnostics["resident_commit_skips"] += 1

    def finish_resident(self, states: dict[int, DeltaNetState]) -> None:
        if not states or not all(state.is_resident for state in states.values()):
            raise ValueError("finish_resident requires resident DeltaNet state views")
        self._record_resident_commit_skip()

    def gather(self, seq_ids: list[int]) -> dict[int, DeltaNetState]:
        """Materialize a reordered state batch as a correctness fallback."""
        self._record_fallback_gather()
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
            gathered_bytes = len(seq_ids) * self.state_bytes_per_sequence
            self._diagnostics["gather_calls"] += 1
            self._diagnostics["gather_layer_ops"] += len(states)
            self._diagnostics["gather_bytes"] += gathered_bytes
            self._diagnostics["slot_id_upload_bytes"] += (
                slot_ids.numel() * slot_ids.element_size()
            )
            self._diagnostics["batch_sizes"].append(len(seq_ids))
        return states

    def commit(self, states: dict[int, DeltaNetState]) -> bool:
        """Commit fallback batches; resident views already updated the pool."""
        if states and all(state.is_resident for state in states.values()):
            self.finish_resident(states)
            return False
        self._record_fallback_commit()
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
        return True

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
