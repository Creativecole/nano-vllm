from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(
    max_num_seqs=2,
    max_num_batched_tokens=64,
    max_prefill_chunk_tokens=4,
    max_partial_prefills=1,
    max_long_partial_prefills=1,
    long_prefill_token_threshold=0,
    decode_reserve_blocks_per_seq=1,
):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
        enable_prefix_cache=False,
        scheduler_policy="prefill_first",
        max_prefill_chunk_tokens=max_prefill_chunk_tokens,
        max_partial_prefills=max_partial_prefills,
        max_long_partial_prefills=max_long_partial_prefills,
        long_prefill_token_threshold=long_prefill_token_threshold,
        decode_reserve_blocks_per_seq=decode_reserve_blocks_per_seq,
    )
    Sequence.block_size = config.kvcache_block_size
    return Scheduler(config)


def make_sequence(tokens, max_tokens):
    return Sequence(
        tokens,
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
    )


def test_dynamic_admission_completion_and_batch_compaction():
    scheduler = make_scheduler(max_num_seqs=2)
    first = make_sequence([1, 2, 3], max_tokens=1)
    second = make_sequence([4, 5], max_tokens=3)
    third = make_sequence([6, 7, 8, 9], max_tokens=2)
    scheduler.add(first)
    scheduler.add(second)
    scheduler.add(third)

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill
    assert scheduled == [first, second]
    scheduler.postprocess(scheduled, [10, 11], is_prefill=True)
    assert first.status is SequenceStatus.FINISHED
    assert second.status is SequenceStatus.RUNNING
    assert scheduler.drain_released_seq_ids() == [first.seq_id]

    scheduled, is_prefill = scheduler.schedule()
    assert is_prefill
    assert scheduled == [third]
    scheduler.postprocess(scheduled, [12], is_prefill=True)
    assert list(scheduler.running) == [second, third]

    scheduled, is_prefill = scheduler.schedule()
    assert not is_prefill
    assert scheduled == [second, third]


def test_prefix_reuse_is_disabled_for_hybrid_scheduler():
    scheduler = make_scheduler(max_num_seqs=2)
    first = make_sequence([1, 2, 3, 4, 5], max_tokens=1)
    second = make_sequence([1, 2, 3, 4, 9], max_tokens=1)
    assert scheduler.block_manager.can_allocate(first) == 0
    scheduler.block_manager.allocate(first, 0)
    first.num_scheduled_tokens = len(first)
    scheduler.block_manager.hash_blocks(first)
    assert scheduler.block_manager.can_allocate(second) == 0


def test_interleaved_scheduler_prioritizes_decode_then_prefill_chunk():
    scheduler = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_prefill_chunk_tokens=3,
    )
    decoding = make_sequence([1, 2], max_tokens=3)
    scheduler.add(decoding)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [10], is_prefill)
    assert decoding.status is SequenceStatus.RUNNING

    prefilling = make_sequence(list(range(20, 30)), max_tokens=2)
    scheduler.add(prefilling)

    decode_batch = scheduler.schedule_decode(token_budget=8)
    assert decode_batch == [decoding]
    scheduler.postprocess(decode_batch, [11], is_prefill=False)

    prefill_batch = scheduler.schedule_prefill(
        token_budget=7,
        chunk_tokens=3,
    )
    assert prefill_batch == [prefilling]
    assert prefilling.num_scheduled_tokens == 3
    assert prefilling.status is SequenceStatus.WAITING
    assert prefilling.prefill_started_at is not None


def test_prefill_chunk_progresses_without_appending_early_tokens():
    scheduler = make_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=8,
        max_prefill_chunk_tokens=3,
    )
    sequence = make_sequence(list(range(10)), max_tokens=2)
    scheduler.add(sequence)

    chunk_sizes = []
    for _ in range(4):
        batch = scheduler.schedule_prefill(8, chunk_tokens=3)
        chunk_sizes.append(batch[0].num_scheduled_tokens)
        previous_tokens = sequence.num_tokens
        scheduler.postprocess(batch, [42], is_prefill=True)
        if sequence.num_cached_tokens < len(sequence.prompt_token_ids):
            assert sequence.num_tokens == previous_tokens

    assert chunk_sizes == [3, 3, 3, 1]
    assert sequence.status is SequenceStatus.RUNNING
    assert sequence.completion_token_ids == [42]


def test_unified_scheduler_builds_one_decode_first_token_budget():
    scheduler = make_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=8,
        max_prefill_chunk_tokens=3,
        max_partial_prefills=2,
        decode_reserve_blocks_per_seq=0,
    )
    decoding = make_sequence([1, 2], max_tokens=3)
    scheduler.add(decoding)
    scheduled, is_prefill = scheduler.schedule()
    scheduler.postprocess(scheduled, [10], is_prefill)

    prefilling = make_sequence(list(range(20, 30)), max_tokens=2)
    scheduler.add(prefilling)
    output = scheduler.schedule_unified()

    assert output.total_num_scheduled_tokens == 4
    assert output.num_decode_requests == 1
    assert output.num_prefill_requests == 1
    assert [item.request_id for item in output.requests] == [
        decoding.seq_id,
        prefilling.seq_id,
    ]
    assert [item.num_scheduled_tokens for item in output.requests] == [1, 3]
    assert [item.sample for item in output.requests] == [True, False]


def test_unified_scheduler_samples_only_final_prefill_chunk():
    scheduler = make_scheduler(
        max_num_seqs=1,
        max_num_batched_tokens=3,
        max_prefill_chunk_tokens=3,
        decode_reserve_blocks_per_seq=0,
    )
    sequence = make_sequence(list(range(7)), max_tokens=2)
    scheduler.add(sequence)

    first = scheduler.schedule_unified()
    assert not first.requests[0].sample
    scheduler.postprocess_unified(first, {})
    second = scheduler.schedule_unified()
    assert not second.requests[0].sample
    scheduler.postprocess_unified(second, {})
    final = scheduler.schedule_unified()
    assert final.requests[0].sample
    scheduler.postprocess_unified(final, {sequence.seq_id: 99})

    assert sequence.status is SequenceStatus.RUNNING
    assert sequence.completion_token_ids == [99]


def test_decode_reservation_prevents_prefill_over_admission():
    scheduler = make_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_prefill_chunk_tokens=4,
        max_partial_prefills=2,
        decode_reserve_blocks_per_seq=1,
    )
    scheduler.block_manager.free_block_ids.clear()
    scheduler.block_manager.free_block_ids.extend(range(4))
    first = make_sequence(list(range(8)), max_tokens=2)
    second = make_sequence(list(range(8, 16)), max_tokens=2)
    scheduler.add(first)
    scheduler.add(second)

    output = scheduler.schedule_unified()

    assert [item.request_id for item in output.requests] == [first.seq_id]
    assert scheduler.block_manager.num_reserved_blocks == 1
    assert scheduler.get_resource_stats()["waiting_requests"] == 2


def test_long_partial_prefill_limit_allows_short_request_to_advance():
    scheduler = make_scheduler(
        max_num_seqs=3,
        max_num_batched_tokens=8,
        max_prefill_chunk_tokens=4,
        max_partial_prefills=2,
        max_long_partial_prefills=1,
        long_prefill_token_threshold=8,
        decode_reserve_blocks_per_seq=0,
    )
    long_first = make_sequence(list(range(20)), max_tokens=2)
    long_second = make_sequence(list(range(20, 40)), max_tokens=2)
    short = make_sequence(list(range(100, 106)), max_tokens=2)
    scheduler.add(long_first)
    scheduler.add(long_second)
    scheduler.add(short)

    output = scheduler.schedule_unified()

    assert [item.request_id for item in output.requests] == [
        long_first.seq_id,
        short.seq_id,
    ]
