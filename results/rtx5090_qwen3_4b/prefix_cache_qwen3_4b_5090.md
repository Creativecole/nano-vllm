# Prefix Cache Benchmark

| workload | prompt_len | max_tokens | ttft_s | decode_tokens_per_s | itl_ms_avg | prefix_cache_hits | prefix_cache_misses | prefix_cache_hit_rate | max_block_utilization | peak_gpu_memory_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_shared_prefix | 512 | 128 | 0.3744 | 115.8689 | 8.6304 | 0 | 4 | 0.0000 | 0.0213 | 27.5493 |
| shared_system_prompt | 512 | 128 | 0.2107 | 122.5501 | 8.1599 | 4 | 0 | 1.0000 | 0.0259 | 27.3754 |
| shared_few_shot_prefix | 512 | 128 | 0.0653 | 168.8295 | 5.9231 | 4 | 0 | 1.0000 | 0.0259 | 27.3754 |
| long_prompt_short_decode | 2048 | 32 | 0.1906 | 166.6695 | 5.9999 | 16 | 4 | 0.8000 | 0.0714 | 26.9535 |
| short_prompt_long_decode | 128 | 256 | 0.0546 | 167.0396 | 5.9866 | 0 | 0 | 0.0000 | 0.0228 | 27.4809 |

## Workload Notes

| workload | note |
| --- | --- |
| no_shared_prefix | Each request starts with a different full block, so prefix-cache hits should stay near zero. |
| shared_system_prompt | One priming request populates shared system-prompt blocks before measured requests arrive. |
| shared_few_shot_prefix | A longer shared prefix should increase prefix-cache hit opportunities. |
| long_prompt_short_decode | Stresses prefill and KV block reuse more than decode. |
| short_prompt_long_decode | Stresses decode and ITL; short prompts may not contain enough full blocks for prefix-cache hits. |
