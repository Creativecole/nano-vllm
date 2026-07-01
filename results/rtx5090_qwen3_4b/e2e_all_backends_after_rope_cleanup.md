# E2E v1/v2 Prompt/Block Sweep

- model: `../models/Qwen3-4B`
- prompt_lens: `512`
- block_sizes: `256`
- num_prompts: `4`
- max_tokens: `128`
- repeat: `3`

| backend | prompt_len | block_size | decode_tokens_per_s | itl_ms_avg | decode_step_ms_p50 | decode_step_ms_p95 |
| --- | --- | --- | --- | --- | --- | --- |
| flash_attn | 512 | 256 | 263.0949 | 3.8033 | 14.8115 | 15.2842 |
| triton_paged_decode | 512 | 256 | 247.6774 | 4.0380 | 16.1016 | 16.5184 |
| triton_paged_decode_v2 | 512 | 256 | 247.7092 | 4.0371 | 16.0557 | 16.5239 |
