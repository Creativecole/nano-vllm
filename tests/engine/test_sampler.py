import torch

from nanovllm.layers.sampler import Sampler


def test_zero_temperature_uses_greedy_tokens():
    logits = torch.tensor([[0.1, 2.0, 0.5], [3.0, -1.0, 2.0]])
    temperatures = torch.zeros(2)

    assert torch.equal(Sampler()(logits, temperatures), torch.tensor([1, 0]))
