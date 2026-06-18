# Upstream vs Fork E2E Benchmark

## Config

| metric | value |
| --- | --- |
| model | ../models/Qwen3-4B |
| prompt_len | 512 |
| num_prompts | 4 |
| max_tokens | 128 |
| repeat | 3 |
| warmup | 1 |
| enforce_eager | True |

## upstream status

| metric | value |
| --- | --- |
| elapsed_s_mean | 3.3062 |
| elapsed_s_p50 | 3.3176 |
| elapsed_s_p95 | 3.3223 |
| ttft_s_mean | 0.1045 |
| ttft_s_p50 | 0.1045 |
| ttft_s_p95 | 0.1062 |
| prefill_time_s_mean | 0.0787 |
| prefill_time_s_p50 | 0.0783 |
| prefill_time_s_p95 | 0.0800 |
| decode_time_s_mean | 3.2246 |
| decode_time_s_p50 | 3.2364 |
| decode_time_s_p95 | 3.2416 |
| total_tokens_per_s_mean | 774.3359 |
| total_tokens_per_s_p50 | 771.6467 |
| total_tokens_per_s_p95 | 780.8027 |
| decode_tokens_per_s_mean | 157.5442 |
| decode_tokens_per_s_p50 | 156.9631 |
| decode_tokens_per_s_p95 | 158.9574 |
| itl_ms_avg_mean | 6.3477 |
| itl_ms_avg_p50 | 6.3709 |
| itl_ms_avg_p95 | 6.3811 |
| decode_step_ms_p50_mean | 25.2246 |
| decode_step_ms_p50_p50 | 25.2274 |
| decode_step_ms_p50_p95 | 25.4185 |
| decode_step_ms_p95_mean | 26.4125 |
| decode_step_ms_p95_p50 | 26.2326 |
| decode_step_ms_p95_p95 | 26.8366 |
| peak_gpu_memory_gb_mean | 27.4288 |
| peak_gpu_memory_gb_p50 | 27.3754 |
| peak_gpu_memory_gb_p95 | 27.5356 |
| gpu | NVIDIA GeForce RTX 5090 |
| num_kvcache_blocks |  |
| max_used_blocks |  |
| max_block_utilization |  |
| kv_cache_dtype |  |
| linear_backend |  |
| norm_backend |  |
| activation_backend |  |
| rope_backend |  |

### Per-run

| run | elapsed_s | ttft_s | decode_tokens_per_s | itl_ms_avg | decode_step_ms_p95 | peak_gpu_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 3.3176 | 0.1045 | 156.9631 | 6.3709 | 26.8366 | 27.3754 |
| 2 | 3.3223 | 0.1029 | 156.7122 | 6.3811 | 26.2326 | 27.3754 |
| 3 | 3.2787 | 0.1062 | 158.9574 | 6.2910 | 26.1683 | 27.5356 |

## fork status

| metric | value |
| --- | --- |
| elapsed_s_mean | 2.4158 |
| elapsed_s_p50 | 2.4089 |
| elapsed_s_p95 | 2.4573 |
| ttft_s_mean | 0.0990 |
| ttft_s_p50 | 0.0990 |
| ttft_s_p95 | 0.0996 |
| prefill_time_s_mean | 0.0797 |
| prefill_time_s_p50 | 0.0797 |
| prefill_time_s_p95 | 0.0797 |
| decode_time_s_mean | 2.3333 |
| decode_time_s_p50 | 2.3264 |
| decode_time_s_p95 | 2.3749 |
| total_tokens_per_s_mean | 1059.8572 |
| total_tokens_per_s_p50 | 1062.7122 |
| total_tokens_per_s_p95 | 1075.0529 |
| decode_tokens_per_s_mean | 217.7530 |
| decode_tokens_per_s_p50 | 218.3592 |
| decode_tokens_per_s_p95 | 220.9923 |
| itl_ms_avg_mean | 4.5932 |
| itl_ms_avg_p50 | 4.5796 |
| itl_ms_avg_p95 | 4.6749 |
| decode_step_ms_p50_mean | 18.2285 |
| decode_step_ms_p50_p50 | 18.1943 |
| decode_step_ms_p50_p95 | 18.5385 |
| decode_step_ms_p95_mean | 19.3240 |
| decode_step_ms_p95_p50 | 19.1777 |
| decode_step_ms_p95_p95 | 19.9104 |
| peak_gpu_memory_gb_mean | 27.3754 |
| peak_gpu_memory_gb_p50 | 27.3754 |
| peak_gpu_memory_gb_p95 | 27.3754 |
| gpu | NVIDIA GeForce RTX 5090 |
| num_kvcache_blocks | 348 |
| max_used_blocks | 12 |
| max_block_utilization | 0.0345 |
| kv_cache_dtype | bfloat16 |
| linear_backend | torch |
| norm_backend | triton |
| activation_backend | triton |
| rope_backend | triton |

### Per-run

| run | elapsed_s | ttft_s | decode_tokens_per_s | itl_ms_avg | decode_step_ms_p95 | peak_gpu_memory_gb |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.3813 | 0.0984 | 220.9923 | 4.5250 | 18.8838 | 27.3754 |
| 2 | 2.4089 | 0.0990 | 218.3592 | 4.5796 | 19.1777 | 27.3754 |
| 3 | 2.4573 | 0.0996 | 213.9076 | 4.6749 | 19.9104 | 27.3754 |

## Direct Comparison

| metric | upstream | fork | fork_vs_upstream |
| --- | --- | --- | --- |
| elapsed_s_mean | 3.3062 | 2.4158 | 1.369x |
| ttft_s_mean | 0.1045 | 0.0990 | 1.056x |
| decode_tokens_per_s_mean | 157.5442 | 217.7530 | 1.382x |
| itl_ms_avg_mean | 6.3477 | 4.5932 | 1.382x |
| decode_step_ms_p95_mean | 26.4125 | 19.3240 | 1.367x |
| peak_gpu_memory_gb_mean | 27.4288 | 27.3754 | 1.002x |
| max_used_blocks |  | 12 |  |
| max_block_utilization |  | 0.0345 |  |



For latency and memory metrics, `fork_vs_upstream` is computed as upstream / fork. For throughput metrics, it is computed as fork / upstream.
