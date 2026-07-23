from collections import deque
from dataclasses import dataclass
from time import perf_counter

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(slots=True)
class ScheduledRequest:
    sequence: Sequence
    num_scheduled_tokens: int
    is_prefill: bool
    sample: bool

    @property
    def request_id(self) -> int:
        return self.sequence.seq_id


@dataclass(slots=True)
class SchedulerOutput:
    requests: list[ScheduledRequest]
    token_budget: int

    @property
    def total_num_scheduled_tokens(self) -> int:
        return sum(item.num_scheduled_tokens for item in self.requests)

    @property
    def num_decode_requests(self) -> int:
        return sum(not item.is_prefill for item in self.requests)

    @property
    def num_prefill_requests(self) -> int:
        return sum(item.is_prefill for item in self.requests)


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.scheduler_policy = getattr(
            config, "scheduler_policy", "prefill_first"
        )
        self.max_prefill_chunk_tokens = getattr(
            config, "max_prefill_chunk_tokens", 256
        )
        self.max_partial_prefills = getattr(config, "max_partial_prefills", 1)
        self.max_long_partial_prefills = getattr(
            config, "max_long_partial_prefills", 1
        )
        self.long_prefill_token_threshold = getattr(
            config, "long_prefill_token_threshold", 0
        )
        self.decode_reserve_blocks_per_seq = getattr(
            config, "decode_reserve_blocks_per_seq", 1
        )
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_cache=config.enable_prefix_cache,
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.released_seq_ids: list[int] = []
        self.step_id = 0
        self.max_waiting_requests = 0
        self.max_running_requests = 0
        self.max_kv_used_blocks = 0
        self.max_kv_reserved_blocks = 0
        self.last_schedule_debug: list[dict[str, int | str]] = []

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self._record_resource_stats()

    def _record_resource_stats(self) -> None:
        self.max_waiting_requests = max(
            self.max_waiting_requests, len(self.waiting)
        )
        self.max_running_requests = max(
            self.max_running_requests, len(self.running)
        )
        self.max_kv_used_blocks = max(
            self.max_kv_used_blocks,
            len(self.block_manager.used_block_ids),
        )
        self.max_kv_reserved_blocks = max(
            self.max_kv_reserved_blocks,
            self.block_manager.num_reserved_blocks,
        )

    def reset_resource_stats(self) -> None:
        self.max_waiting_requests = len(self.waiting)
        self.max_running_requests = len(self.running)
        self.max_kv_used_blocks = len(self.block_manager.used_block_ids)
        self.max_kv_reserved_blocks = self.block_manager.num_reserved_blocks
        self.last_schedule_debug = []

    def drain_released_seq_ids(self) -> list[int]:
        released, self.released_seq_ids = self.released_seq_ids, []
        return released

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        available_prefill_slots = self.max_num_seqs - len(self.running)
        while self.waiting and len(scheduled_seqs) < available_prefill_slots:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            if seq.prefill_started_at is None:
                seq.prefill_started_at = perf_counter()
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def schedule_decode(self, token_budget: int) -> list[Sequence]:
        """Schedule one decode token per running request."""
        if token_budget <= 0:
            return []
        scheduled_seqs = []
        limit = min(self.max_num_seqs, token_budget)
        while self.running and len(scheduled_seqs) < limit:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def schedule_prefill(
        self,
        token_budget: int,
        *,
        chunk_tokens: int | None = None,
    ) -> list[Sequence]:
        """Schedule bounded prefill chunks in the remaining token budget."""
        if token_budget <= 0:
            return []
        chunk_tokens = chunk_tokens or self.max_prefill_chunk_tokens
        scheduled_seqs = []
        num_batched_tokens = 0
        available_slots = self.max_num_seqs - len(self.running)
        while self.waiting and len(scheduled_seqs) < available_slots:
            remaining_budget = token_budget - num_batched_tokens
            if remaining_budget <= 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = (
                    seq.num_tokens - num_cached_blocks * self.block_size
                )
                self.block_manager.allocate(seq, num_cached_blocks)
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            seq.num_scheduled_tokens = min(
                num_tokens,
                remaining_budget,
                chunk_tokens,
            )
            if seq.prefill_started_at is None:
                seq.prefill_started_at = perf_counter()
            num_batched_tokens += seq.num_scheduled_tokens
            if (
                seq.num_cached_tokens + seq.num_scheduled_tokens
                == seq.num_tokens
            ):
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            else:
                # A partially prefilling sequence remains at waiting[0].
                # Schedule it only once in this engine step.
                scheduled_seqs.append(seq)
                break
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def _is_long_prefill(self, seq: Sequence) -> bool:
        threshold = self.long_prefill_token_threshold
        return threshold > 0 and seq.num_prompt_tokens > threshold

    def schedule_unified(self) -> SchedulerOutput:
        """Build one decode-first mixed batch under a shared token budget."""
        self.step_id += 1
        token_budget = self.max_num_batched_tokens
        scheduled: list[ScheduledRequest] = []

        decode_seqs = self.schedule_decode(token_budget)
        for seq in decode_seqs:
            scheduled.append(
                ScheduledRequest(
                    sequence=seq,
                    num_scheduled_tokens=1,
                    is_prefill=False,
                    sample=True,
                )
            )
        token_budget -= len(decode_seqs)

        partial = [
            seq
            for seq in self.waiting
            if seq.block_table and seq.num_cached_tokens < seq.num_tokens
        ]
        active_partial_ids = {seq.seq_id for seq in partial}
        partial_count = len(partial)
        long_partial_count = sum(self._is_long_prefill(seq) for seq in partial)
        active_count = len(self.running) + partial_count

        for seq in list(self.waiting):
            if token_budget <= 0:
                break
            already_partial = seq.seq_id in active_partial_ids
            is_long = self._is_long_prefill(seq)
            if not already_partial:
                if active_count >= self.max_num_seqs:
                    continue
                if partial_count >= self.max_partial_prefills:
                    continue
                if is_long and long_partial_count >= self.max_long_partial_prefills:
                    continue

            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(
                    seq,
                    reserve_blocks=self.decode_reserve_blocks_per_seq,
                )
                if num_cached_blocks == -1:
                    continue
                self.block_manager.allocate(
                    seq,
                    num_cached_blocks,
                    reserve_blocks=self.decode_reserve_blocks_per_seq,
                )
                active_count += 1
                partial_count += 1
                if is_long:
                    long_partial_count += 1
            else:
                num_cached_blocks = seq.num_cached_tokens // self.block_size

            target_num_tokens = seq.num_tokens
            remaining_tokens = target_num_tokens - seq.num_cached_tokens
            if remaining_tokens <= 0:
                continue
            num_scheduled_tokens = min(
                remaining_tokens,
                self.max_prefill_chunk_tokens,
                token_budget,
            )
            seq.num_scheduled_tokens = num_scheduled_tokens
            if seq.prefill_started_at is None:
                seq.prefill_started_at = perf_counter()
            completes_prompt = (
                seq.num_cached_tokens + num_scheduled_tokens
                == target_num_tokens
            )
            if completes_prompt:
                seq.status = SequenceStatus.RUNNING
                self.waiting.remove(seq)
                self.running.append(seq)
                partial_count -= 1
                if is_long:
                    long_partial_count -= 1
            scheduled.append(
                ScheduledRequest(
                    sequence=seq,
                    num_scheduled_tokens=num_scheduled_tokens,
                    is_prefill=True,
                    sample=completes_prompt,
                )
            )
            token_budget -= num_scheduled_tokens

        if not scheduled:
            raise RuntimeError("Unified scheduler produced no work")
        self.last_schedule_debug = [
            {
                "step_id": self.step_id,
                "request_id": item.request_id,
                "scheduled_tokens": item.num_scheduled_tokens,
                "kind": "prefill" if item.is_prefill else "decode",
            }
            for item in scheduled
        ]
        self._record_resource_stats()
        return SchedulerOutput(
            requests=scheduled,
            token_budget=self.max_num_batched_tokens,
        )

    def postprocess_unified(
        self,
        scheduler_output: SchedulerOutput,
        sampled_token_ids: dict[int, int],
    ) -> None:
        for item in scheduler_output.requests:
            seq = item.sequence
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += item.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if not item.sample:
                continue
            if seq.seq_id not in sampled_token_ids:
                raise RuntimeError(
                    f"Missing sampled token for sequence {seq.seq_id}"
                )
            token_id = sampled_token_ids[seq.seq_id]
            seq.append_token(token_id)
            if (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            ):
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.released_seq_ids.append(seq.seq_id)
                self.running.remove(seq)
        self._record_resource_stats()

    def get_resource_stats(self) -> dict[str, object]:
        total_blocks = len(self.block_manager.blocks)
        used_blocks = len(self.block_manager.used_block_ids)
        free_blocks = len(self.block_manager.free_block_ids)
        reserved_blocks = self.block_manager.num_reserved_blocks
        return {
            "waiting_requests": len(self.waiting),
            "running_requests": len(self.running),
            "kv_total_blocks": total_blocks,
            "kv_used_blocks": used_blocks,
            "kv_free_blocks": free_blocks,
            "kv_reserved_blocks": reserved_blocks,
            "kv_block_utilization": used_blocks / total_blocks
            if total_blocks
            else 0.0,
            "max_waiting_requests": self.max_waiting_requests,
            "max_running_requests": self.max_running_requests,
            "max_kv_used_blocks": self.max_kv_used_blocks,
            "max_kv_reserved_blocks": self.max_kv_reserved_blocks,
            "max_kv_block_utilization": self.max_kv_used_blocks / total_blocks
            if total_blocks
            else 0.0,
            "last_schedule": list(self.last_schedule_debug),
        }

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.released_seq_ids.append(seq.seq_id)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.released_seq_ids.append(seq.seq_id)
                self.running.remove(seq)
