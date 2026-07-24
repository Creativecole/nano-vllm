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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen35_decode_fast_path_matches_normal_decode():
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
        max_tokens=6,
        ignore_eos=True,
    )

    def generate(decode_fast_path):
        llm = LLM(
            model_path,
            enforce_eager=True,
            max_num_seqs=2,
            hybrid_state_capacity=2,
            deltanet_backend="chunked",
            resident_deltanet_state=True,
            decode_fast_path=decode_fast_path,
        )
        try:
            request_ids = []
            for prompt in prompts:
                request_ids.append(llm.add_request(prompt, params))
            tokens = {}
            logits = {}
            while not llm.is_finished():
                step_outputs, _, step_logits = llm.step_with_logits()
                tokens.update(dict(step_outputs))
                for seq_id, value in step_logits.items():
                    logits.setdefault(seq_id, []).append(value)
            stats = llm.model_runner.call("get_execution_stats")
            ordered_tokens = [tokens[seq_id] for seq_id in request_ids]
            ordered_logits = [logits[seq_id] for seq_id in request_ids]
            return ordered_tokens, ordered_logits, stats
        finally:
            llm.exit()

    baseline_tokens, baseline_logits, baseline_stats = generate(False)
    fast_tokens, fast_logits, fast_stats = generate(True)

    assert fast_tokens == baseline_tokens
    assert len(fast_logits) == len(baseline_logits)
    for fast_steps, baseline_steps in zip(fast_logits, baseline_logits):
        assert len(fast_steps) == len(baseline_steps)
        for fast, baseline in zip(
            fast_steps,
            baseline_steps,
        ):
            torch.testing.assert_close(fast, baseline, rtol=0, atol=0)
    assert baseline_stats["decode_fast_path_hits"] == 0
    assert fast_stats["decode_fast_path_hits"] > 0
    assert fast_stats["decode_fast_path_builds"] > 0
