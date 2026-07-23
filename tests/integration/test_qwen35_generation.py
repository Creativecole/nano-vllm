import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen35_hybrid_continuous_batching_smoke():
    model_path = os.environ.get("NANOVLLM_QWEN35_MODEL")
    if not model_path:
        pytest.skip("set NANOVLLM_QWEN35_MODEL to run the Qwen3.5 integration test")

    from nanovllm import LLM, SamplingParams

    llm = LLM(
        model_path,
        enforce_eager=True,
        max_num_seqs=4,
        hybrid_state_capacity=4,
    )
    try:
        short = SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True)
        long = SamplingParams(temperature=0.0, max_tokens=4, ignore_eos=True)
        llm.add_request([1, 2, 3, 4], long)
        llm.add_request([5, 6, 7], short)

        # Admit a request after the first prefill to exercise dynamic batch growth.
        outputs = {}
        first_outputs, _ = llm.step()
        outputs.update(first_outputs)
        llm.add_request([8, 9, 10, 11, 12], long)
        while not llm.is_finished():
            step_outputs, _ = llm.step()
            outputs.update(step_outputs)
    finally:
        llm.exit()

    assert sorted(len(tokens) for tokens in outputs.values()) == [2, 4, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen35_interleaved_prefill_matches_greedy_baseline():
    model_path = os.environ.get("NANOVLLM_QWEN35_MODEL")
    if not model_path:
        pytest.skip("set NANOVLLM_QWEN35_MODEL to run the Qwen3.5 integration test")

    from nanovllm import LLM, SamplingParams

    prompts = [
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12],
    ]
    params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        ignore_eos=True,
    )

    def generate(scheduler_policy):
        llm = LLM(
            model_path,
            enforce_eager=True,
            max_num_seqs=2,
            hybrid_state_capacity=2,
            deltanet_backend="chunked",
            scheduler_policy=scheduler_policy,
            max_prefill_chunk_tokens=3,
        )
        try:
            return [
                output["token_ids"]
                for output in llm.generate(
                    prompts,
                    params,
                    use_tqdm=False,
                )
            ]
        finally:
            llm.exit()

    baseline = generate("prefill_first")
    assert generate("interleave") == baseline
    assert generate("unified") == baseline


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen35_resident_state_matches_gather_commit_tokens():
    model_path = os.environ.get("NANOVLLM_QWEN35_MODEL")
    if not model_path:
        pytest.skip("set NANOVLLM_QWEN35_MODEL to run the Qwen3.5 integration test")

    from nanovllm import LLM, SamplingParams

    prompts = [
        [1, 2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12],
        [13, 14, 15, 16, 17, 18],
    ]
    params = [
        SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        for max_tokens in (2, 4, 3)
    ]

    def generate(resident_deltanet_state):
        llm = LLM(
            model_path,
            enforce_eager=True,
            max_num_seqs=3,
            hybrid_state_capacity=3,
            deltanet_backend="chunked",
            resident_deltanet_state=resident_deltanet_state,
        )
        try:
            return [
                output["token_ids"]
                for output in llm.generate(
                    prompts,
                    params,
                    use_tqdm=False,
                )
            ]
        finally:
            llm.exit()

    statecopy_tokens = generate(False)
    assert generate(True) == statecopy_tokens
