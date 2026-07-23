from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    is_mixed: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    prefill_seq_lens: tuple[int, ...] | None = None
    num_decode_requests: int = 0
    decode_context_lens: torch.Tensor | None = None
    decode_block_tables: torch.Tensor | None = None
    prefill_cu_seqlens_q: torch.Tensor | None = None
    prefill_cu_seqlens_k: torch.Tensor | None = None
    prefill_max_seqlen_q: int = 0
    prefill_max_seqlen_k: int = 0
    prefill_block_tables: torch.Tensor | None = None
    sequence_query_lens: tuple[int, ...] | None = None
    sample_indices: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    prefill_seq_lens=None,
    *,
    is_mixed=False,
    num_decode_requests=0,
    decode_context_lens=None,
    decode_block_tables=None,
    prefill_cu_seqlens_q=None,
    prefill_cu_seqlens_k=None,
    prefill_max_seqlen_q=0,
    prefill_max_seqlen_k=0,
    prefill_block_tables=None,
    sequence_query_lens=None,
    sample_indices=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill=is_prefill,
        is_mixed=is_mixed,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        prefill_seq_lens=prefill_seq_lens,
        num_decode_requests=num_decode_requests,
        decode_context_lens=decode_context_lens,
        decode_block_tables=decode_block_tables,
        prefill_cu_seqlens_q=prefill_cu_seqlens_q,
        prefill_cu_seqlens_k=prefill_cu_seqlens_k,
        prefill_max_seqlen_q=prefill_max_seqlen_q,
        prefill_max_seqlen_k=prefill_max_seqlen_k,
        prefill_block_tables=prefill_block_tables,
        sequence_query_lens=sequence_query_lens,
        sample_indices=sample_indices,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
