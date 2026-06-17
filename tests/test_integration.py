import os

import pytest
import torch


pytest.importorskip("flash_attn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif("NANOVLLM_TEST_MODEL" not in os.environ, reason="set NANOVLLM_TEST_MODEL to a local model path")
def test_bf16_generate():
    from nanovllm import LLM, SamplingParams

    model = os.environ["NANOVLLM_TEST_MODEL"]
    prompts = ["Hello, Nano-vLLM."]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=8)
    llm = LLM(model, enforce_eager=True, max_model_len=512)
    try:
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    finally:
        llm.exit()

    assert len(outputs) == 1
    assert len(outputs[0]["token_ids"]) > 0
