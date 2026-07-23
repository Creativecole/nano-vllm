from types import SimpleNamespace

import torch

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler import ScheduledRequest, SchedulerOutput
from nanovllm.sampling_params import SamplingParams


class FakeScheduler:
    def __init__(self, seqs):
        self.seqs = seqs
        self.release_batches = [[99], [seq.seq_id for seq in seqs]]

    def add(self, seq):
        self.seqs.append(seq)

    def schedule(self):
        return self.seqs, False

    def drain_released_seq_ids(self):
        return self.release_batches.pop(0)

    def postprocess(self, seqs, token_ids, is_prefill):
        assert not is_prefill
        for seq, token_id in zip(seqs, token_ids):
            seq.completion_token_ids = [token_id]
            seq.is_finished = True


class FakeRunner:
    def __init__(self):
        self.releases = []

    def call(self, method, *args):
        if method == "release_states":
            self.releases.append(args[0])
            return None
        if method == "run":
            assert args[-1] is True
            return [3, 4], torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        raise AssertionError(method)


class FakeInterleavedScheduler:
    def __init__(self, decode_seq, prefill_seq):
        self.decode_seq = decode_seq
        self.prefill_seq = prefill_seq
        self.prefill_budget = None

    def schedule_decode(self, token_budget):
        assert token_budget == 8
        return [self.decode_seq]

    def schedule_prefill(self, token_budget, *, chunk_tokens):
        self.prefill_budget = token_budget
        assert chunk_tokens == 3
        return [self.prefill_seq]

    def drain_released_seq_ids(self):
        return []

    def postprocess(self, seqs, token_ids, is_prefill):
        if is_prefill:
            self.prefill_seq.num_cached_tokens += (
                self.prefill_seq.num_scheduled_tokens
            )
        else:
            self.decode_seq.completion_token_ids.extend(token_ids)


class FakeInterleavedRunner:
    def __init__(self):
        self.run_calls = []

    def call(self, method, *args):
        assert method == "run"
        self.run_calls.append(args)
        is_prefill = args[1]
        skip_sampling = args[3]
        if is_prefill:
            assert skip_sampling
            return [0]
        assert not skip_sampling
        return [7]


def test_add_request_returns_stable_sequence_id():
    engine = LLMEngine.__new__(LLMEngine)
    engine.tokenizer = SimpleNamespace(encode=lambda prompt: [1, 2])
    engine.scheduler = SimpleNamespace(add=lambda seq: setattr(engine, "added", seq))
    seq_id = engine.add_request("hello", SamplingParams())
    assert seq_id == engine.added.seq_id


def test_step_with_logits_uses_normal_release_and_postprocess_path():
    seqs = [
        SimpleNamespace(seq_id=10, completion_token_ids=[], is_finished=False),
        SimpleNamespace(seq_id=20, completion_token_ids=[], is_finished=False),
    ]
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = FakeScheduler(seqs)
    engine.model_runner = FakeRunner()

    outputs, num_tokens, logits = engine.step_with_logits()

    assert num_tokens == -2
    assert [seq_id for seq_id, _ in outputs] == [10, 20]
    assert torch.equal(logits[10], torch.tensor([0.1, 0.2]))
    assert engine.model_runner.releases == [[99], [10, 20]]


def test_interleaved_step_runs_decode_before_bounded_prefill():
    decode_seq = SimpleNamespace(
        seq_id=10,
        num_cached_tokens=4,
        num_scheduled_tokens=1,
        num_tokens=5,
        completion_token_ids=[],
        is_finished=False,
    )
    prefill_seq = SimpleNamespace(
        seq_id=20,
        num_cached_tokens=0,
        num_scheduled_tokens=3,
        num_tokens=10,
        completion_token_ids=[],
        is_finished=False,
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        scheduler_policy="interleave",
        max_num_batched_tokens=8,
        max_prefill_chunk_tokens=3,
    )
    engine.scheduler = FakeInterleavedScheduler(decode_seq, prefill_seq)
    engine.model_runner = FakeInterleavedRunner()

    outputs, num_tokens = engine.step()

    assert outputs == []
    assert num_tokens == -1
    assert engine.scheduler.prefill_budget == 3
    assert [call[1] for call in engine.model_runner.run_calls] == [False, True]
    assert prefill_seq.num_cached_tokens == 3


class FakeUnifiedScheduler:
    def __init__(self, output):
        self.output = output
        self.sampled = None

    def schedule_unified(self):
        return self.output

    def drain_released_seq_ids(self):
        return []

    def postprocess_unified(self, output, sampled):
        assert output is self.output
        self.sampled = sampled


class FakeUnifiedRunner:
    def __init__(self):
        self.calls = []

    def call(self, method, *args):
        self.calls.append((method, args))
        assert method == "run_mixed"
        return {10: 7}


def test_unified_step_uses_one_model_runner_call():
    decode_seq = SimpleNamespace(
        seq_id=10,
        completion_token_ids=[],
        is_finished=False,
    )
    prefill_seq = SimpleNamespace(
        seq_id=20,
        completion_token_ids=[],
        is_finished=False,
    )
    output = SchedulerOutput(
        requests=[
            ScheduledRequest(decode_seq, 1, False, True),
            ScheduledRequest(prefill_seq, 3, True, False),
        ],
        token_budget=8,
    )
    engine = LLMEngine.__new__(LLMEngine)
    engine.config = SimpleNamespace(scheduler_policy="unified")
    engine.scheduler = FakeUnifiedScheduler(output)
    engine.model_runner = FakeUnifiedRunner()

    outputs, num_tokens = engine.step()

    assert outputs == []
    assert num_tokens == -1
    assert len(engine.model_runner.calls) == 1
    assert engine.scheduler.sampled == {10: 7}
