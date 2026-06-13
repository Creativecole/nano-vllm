import os

import pytest
import torch


pytest.importorskip("triton")
pytest.importorskip("flash_attn")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif("NANOVLLM_TEST_MODEL" not in os.environ, reason="set NANOVLLM_TEST_MODEL to a local model path")
def test_bf16_and_fp8_kv_generate():
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.attention import flash_attn_supports_fp8_kvcache

    model = os.environ["NANOVLLM_TEST_MODEL"]
    prompts = ["Hello, Nano-vLLM."]
    sampling_params = SamplingParams(temperature=1.0, max_tokens=8)
    for dtype in ["bf16", "fp8_e4m3"]:
        if dtype == "fp8_e4m3" and not flash_attn_supports_fp8_kvcache():
            pytest.skip("FlashAttention build does not expose FP8 k_descale/v_descale")
        llm = LLM(model, enforce_eager=True, max_model_len=512, kv_cache_dtype=dtype)
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        llm.exit()
        assert len(outputs) == 1
        assert len(outputs[0]["token_ids"]) > 0
