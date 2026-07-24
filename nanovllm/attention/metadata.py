from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class CommonExecutionMetadata:
    """Per-step metadata shared by all sequence mixer backends."""

    request_ids: tuple[int, ...]
    query_lens: tuple[int, ...]
    is_prefilling: tuple[bool, ...]
    num_requests: int
    num_tokens: int
    positions: torch.Tensor


@dataclass(slots=True)
class FullAttentionMetadata:
    """Metadata consumed by the paged full-attention backend."""

    common: CommonExecutionMetadata
    is_prefill: bool
    is_mixed: bool
    slot_mapping: torch.Tensor | None
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    num_decode_requests: int = 0
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    prefill_cu_seqlens_q: torch.Tensor | None = None
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_max_seqlen_q: int = 0
    prefill_max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None


@dataclass(slots=True)
class DeltaNetMetadata:
    """Metadata consumed by recurrent/linear-attention backends."""

    common: CommonExecutionMetadata
    is_prefill: bool
    is_mixed: bool
    prefill_seq_lens: tuple[int, ...] | None = None
    sequence_query_lens: tuple[int, ...] | None = None
    num_decode_requests: int = 0


@dataclass(slots=True)
class HybridAttentionMetadata:
    """Typed metadata bundle for a hybrid model execution step."""

    common: CommonExecutionMetadata
    full_attention: FullAttentionMetadata
    deltanet: DeltaNetMetadata

    def for_layer_type(self, layer_type: str):
        if layer_type == "full_attention":
            return self.full_attention
        if layer_type == "linear_attention":
            return self.deltanet
        raise KeyError(f"No hybrid metadata registered for layer type {layer_type!r}")


class HybridAttentionMetadataBuilder:
    """Build typed backend metadata from one scheduler/model-runner step."""

    def build(
        self,
        *,
        context,
        request_ids: list[int],
        query_lens: list[int] | tuple[int, ...],
        is_prefilling: list[bool] | tuple[bool, ...],
        positions: torch.Tensor,
    ) -> HybridAttentionMetadata:
        return build_hybrid_attention_metadata(
            context=context,
            request_ids=request_ids,
            query_lens=query_lens,
            is_prefilling=is_prefilling,
            positions=positions,
        )


def build_hybrid_attention_metadata(
    *,
    context,
    request_ids: list[int],
    query_lens: list[int] | tuple[int, ...],
    is_prefilling: list[bool] | tuple[bool, ...],
    positions: torch.Tensor,
) -> HybridAttentionMetadata:
    """Adapt the legacy runtime context into typed backend metadata.

    The adapter lets the model/backend path stop depending on the global context
    while the non-hybrid Qwen3 path remains unchanged during migration.
    """

    request_ids_tuple = tuple(request_ids)
    query_lens_tuple = tuple(query_lens)
    is_prefilling_tuple = tuple(is_prefilling)
    if not (
        len(request_ids_tuple)
        == len(query_lens_tuple)
        == len(is_prefilling_tuple)
    ):
        raise ValueError(
            "Hybrid metadata request_ids, query_lens, and is_prefilling must "
            "have identical lengths"
        )
    if sum(query_lens_tuple) != positions.numel():
        raise ValueError(
            "Hybrid metadata query lengths do not match packed token count: "
            f"query_tokens={sum(query_lens_tuple)}, positions={positions.numel()}"
        )

    common = CommonExecutionMetadata(
        request_ids=request_ids_tuple,
        query_lens=query_lens_tuple,
        is_prefilling=is_prefilling_tuple,
        num_requests=len(request_ids_tuple),
        num_tokens=positions.numel(),
        positions=positions,
    )
    full_attention = FullAttentionMetadata(
        common=common,
        is_prefill=context.is_prefill,
        is_mixed=context.is_mixed,
        slot_mapping=context.slot_mapping,
        cu_seqlens_q=context.cu_seqlens_q,
        cu_seqlens_k=context.cu_seqlens_k,
        max_seqlen_q=context.max_seqlen_q,
        max_seqlen_k=context.max_seqlen_k,
        context_lens=context.context_lens,
        block_tables=context.block_tables,
        num_decode_requests=context.num_decode_requests,
        decode_context_lens=context.decode_context_lens,
        decode_block_tables=context.decode_block_tables,
        prefill_cu_seqlens_q=context.prefill_cu_seqlens_q,
        prefill_cu_seqlens_k=context.prefill_cu_seqlens_k,
        prefill_max_seqlen_q=context.prefill_max_seqlen_q,
        prefill_max_seqlen_k=context.prefill_max_seqlen_k,
        prefill_block_tables=context.prefill_block_tables,
    )
    deltanet = DeltaNetMetadata(
        common=common,
        is_prefill=context.is_prefill,
        is_mixed=context.is_mixed,
        prefill_seq_lens=context.prefill_seq_lens,
        sequence_query_lens=context.sequence_query_lens,
        num_decode_requests=context.num_decode_requests,
    )
    return HybridAttentionMetadata(
        common=common,
        full_attention=full_attention,
        deltanet=deltanet,
    )
