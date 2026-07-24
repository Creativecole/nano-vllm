from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from nanovllm.utils.context import set_context


@dataclass(slots=True)
class HybridDecodeContext:
    """Reusable host/runtime structure for a stable resident decode batch.

    KV and DeltaNet payloads remain owned by their cache managers. This object
    retains only request/layout identity, resident state views, metadata buffers,
    and the previous device-side sampled tokens.
    """

    mode: str
    requested_ids: tuple[int, ...]
    execution_ids: tuple[int, ...]
    execution_rows: tuple[int, ...]
    sequence_lengths: tuple[int, ...]
    block_table_lengths: tuple[int, ...]
    last_block_ids: tuple[int, ...]
    block_table_signature: tuple[tuple[int, ...], ...]
    sampling_signature: tuple[float, ...]
    state_layout_version: int
    layer_states: dict[int, Any]
    attention_metadata: Any
    temperatures: torch.Tensor
    positions: torch.Tensor
    slot_mapping: torch.Tensor
    context_lens: torch.Tensor
    block_tables: torch.Tensor
    sample_indices: torch.Tensor | None = None
    next_input_ids: torch.Tensor | None = None
    valid: bool = True
    invalidation_reason: str | None = None

    @classmethod
    def create(
        cls,
        *,
        mode: str,
        requested_seqs,
        execution_seqs,
        state_layout_version: int,
        layer_states: dict[int, Any],
        attention_metadata,
        temperatures: torch.Tensor,
    ) -> "HybridDecodeContext":
        if mode not in ("decode", "unified_decode"):
            raise ValueError(f"Unsupported decode context mode {mode!r}")
        requested_ids = tuple(seq.seq_id for seq in requested_seqs)
        execution_ids = tuple(seq.seq_id for seq in execution_seqs)
        requested_row = {
            seq_id: row for row, seq_id in enumerate(requested_ids)
        }
        execution_rows = tuple(requested_row[seq_id] for seq_id in execution_ids)
        full_metadata = attention_metadata.full_attention
        context_lens = (
            full_metadata.context_lens
            if mode == "decode"
            else full_metadata.decode_context_lens
        )
        block_tables = (
            full_metadata.block_tables
            if mode == "decode"
            else full_metadata.decode_block_tables
        )
        if (
            full_metadata.slot_mapping is None
            or context_lens is None
            or block_tables is None
        ):
            raise ValueError("Decode context requires slot, length, and block metadata")
        if any(not seq.block_table for seq in execution_seqs):
            raise ValueError("Decode context requires allocated KV block tables")

        return cls(
            mode=mode,
            requested_ids=requested_ids,
            execution_ids=execution_ids,
            execution_rows=execution_rows,
            sequence_lengths=tuple(len(seq) for seq in execution_seqs),
            block_table_lengths=tuple(
                len(seq.block_table) for seq in execution_seqs
            ),
            last_block_ids=tuple(seq.block_table[-1] for seq in execution_seqs),
            block_table_signature=tuple(
                tuple(seq.block_table) for seq in execution_seqs
            ),
            sampling_signature=tuple(
                float(seq.temperature) for seq in execution_seqs
            ),
            state_layout_version=state_layout_version,
            layer_states=layer_states,
            attention_metadata=attention_metadata,
            temperatures=temperatures,
            positions=attention_metadata.common.positions,
            slot_mapping=full_metadata.slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            sample_indices=(
                torch.arange(
                    len(execution_seqs),
                    dtype=torch.long,
                    device=full_metadata.common.positions.device,
                )
                if mode == "unified_decode"
                else None
            ),
        )

    def invalidate(self, reason: str) -> None:
        self.valid = False
        self.invalidation_reason = reason
        self.next_input_ids = None

    def reuse_failure(
        self,
        requested_seqs,
        *,
        mode: str,
        state_layout_version: int,
    ) -> str | None:
        if not self.valid:
            return self.invalidation_reason or "invalid"
        if mode != self.mode:
            return "mode_changed"
        requested_ids = tuple(seq.seq_id for seq in requested_seqs)
        if requested_ids != self.requested_ids:
            return "batch_changed"
        if state_layout_version != self.state_layout_version:
            return "state_layout_changed"
        if self.next_input_ids is None:
            return "missing_device_tokens"

        execution_seqs = self.execution_sequences(requested_seqs)
        if tuple(len(seq) for seq in execution_seqs) != tuple(
            length + 1 for length in self.sequence_lengths
        ):
            return "sequence_progress_changed"
        if tuple(len(seq.block_table) for seq in execution_seqs) != (
            self.block_table_lengths
        ):
            return "kv_block_layout_changed"
        if tuple(seq.block_table[-1] for seq in execution_seqs) != (
            self.last_block_ids
        ):
            return "kv_block_layout_changed"
        if tuple(tuple(seq.block_table) for seq in execution_seqs) != (
            self.block_table_signature
        ):
            return "kv_block_layout_changed"
        if tuple(float(seq.temperature) for seq in execution_seqs) != (
            self.sampling_signature
        ):
            return "sampling_changed"
        return None

    def execution_sequences(self, requested_seqs):
        return [requested_seqs[row] for row in self.execution_rows]

    def advance(self, execution_seqs) -> tuple[torch.Tensor, torch.Tensor]:
        if self.next_input_ids is None:
            raise RuntimeError("Decode context has no sampled device tokens")
        self.positions.add_(1)
        self.context_lens.add_(1)
        self.slot_mapping.add_(1)
        self.sequence_lengths = tuple(len(seq) for seq in execution_seqs)
        return self.next_input_ids, self.positions

    def record_sampled_tokens(self, token_ids: torch.Tensor) -> None:
        if token_ids.ndim != 1 or token_ids.numel() != len(self.execution_ids):
            raise ValueError(
                "Decode sampled-token shape does not match cached execution batch: "
                f"tokens={tuple(token_ids.shape)}, batch={len(self.execution_ids)}"
            )
        self.next_input_ids = token_ids.detach()

    def activate_runtime_context(self) -> None:
        if self.mode == "decode":
            set_context(
                False,
                slot_mapping=self.slot_mapping,
                context_lens=self.context_lens,
                block_tables=self.block_tables,
            )
            return
        set_context(
            False,
            slot_mapping=self.slot_mapping,
            is_mixed=True,
            num_decode_requests=len(self.execution_ids),
            decode_context_lens=self.context_lens,
            decode_block_tables=self.block_tables,
            sequence_query_lens=(1,) * len(self.execution_ids),
            sample_indices=self.sample_indices,
        )
