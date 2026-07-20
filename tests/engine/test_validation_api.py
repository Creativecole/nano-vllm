from types import SimpleNamespace

import torch

from nanovllm.engine.llm_engine import LLMEngine
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
