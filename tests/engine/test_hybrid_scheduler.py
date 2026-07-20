from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def make_scheduler(max_num_seqs=2):
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=64,
        eos=-1,
        kvcache_block_size=4,
        num_kvcache_blocks=32,
        enable_prefix_cache=False,
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
