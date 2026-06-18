## E2E Config

| Metric | Value |
| --- | --- |
| model | ../models/Qwen3-4B |
| gpu | NVIDIA GeForce RTX 5090 |
| prompt_len | 512 |
| num_prompts | 4 |
| max_tokens | 128 |
| kv_cache_dtype | bfloat16 |
| linear_backend | torch |
| norm_backend | triton |
| activation_backend | triton |
| rope_backend | triton |

## Per-Run Results

| run | elapsed_s | ttft_s | prefill_time_s | decode_time_s | decode_tokens_per_s | itl_ms_avg | decode_step_ms_p50 | decode_step_ms_p95 | peak_gpu_memory_gb | max_used_blocks | max_block_utilization |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.4177 | 0.0987 | 0.0797 | 2.3352 | 217.5371 | 4.5969 | 18.2255 | 19.0399 | 27.3754 | 12 | 0.0345 |
| 2 | 2.4344 | 0.0991 | 0.0800 | 2.3516 | 216.0226 | 4.6291 | 18.3836 | 19.3028 | 27.3754 | 12 | 0.0345 |
| 3 | 2.1652 | 0.0958 | 0.0789 | 2.0837 | 243.7994 | 4.1017 | 16.3608 | 16.8242 | 27.3754 | 12 | 0.0345 |

## Aggregate Results

| metric | mean | p50 | p95 |
| --- | --- | --- | --- |
| elapsed_s | 2.3391 | 2.4177 | 2.4344 |
| ttft_s | 0.0979 | 0.0987 | 0.0991 |
| prefill_time_s | 0.0796 | 0.0797 | 0.0800 |
| decode_time_s | 2.2568 | 2.3352 | 2.3516 |
| decode_tokens_per_s | 225.7864 | 217.5371 | 243.7994 |
| itl_ms_avg | 4.4426 | 4.5969 | 4.6291 |
| decode_step_ms_p50 | 17.6566 | 18.2255 | 18.3836 |
| decode_step_ms_p95 | 18.3890 | 19.0399 | 19.3028 |
| peak_gpu_memory_gb | 27.3754 | 27.3754 | 27.3754 |
| max_used_blocks | 12 | 12 | 12 |
| max_block_utilization | 0.0345 | 0.0345 | 0.0345 |
