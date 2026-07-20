import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_qwen3_generation_smoke():
    model_path = os.environ.get("NANOVLLM_QWEN3_MODEL")
    if not model_path:
        pytest.skip("set NANOVLLM_QWEN3_MODEL to run the Qwen3 regression")

    from nanovllm import LLM, SamplingParams

    llm = LLM(model_path, enforce_eager=True, max_num_seqs=2)
    try:
        outputs = llm.generate(
            [[1, 2, 3], [4, 5, 6]],
            SamplingParams(temperature=0.0, max_tokens=2, ignore_eos=True),
            use_tqdm=False,
        )
    finally:
        llm.exit()
    assert [len(output["token_ids"]) for output in outputs] == [2, 2]
