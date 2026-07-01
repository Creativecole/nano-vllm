# nano-vLLM E2E PyTorch Profiler

| Metric | Value |
|---|---:|
| model | ../models/Qwen3-4B |
| gpu | NVIDIA GeForce RTX 5090 |
| attn_backend | triton_paged_decode_v2 |
| block_size | 256 |
| auto_threshold | 1024 |
| prompt_len | 512 |
| num_prompts | 4 |
| max_tokens | 64 |
| profiled_steps | 64 |
| prefill_steps | 1 |
| decode_steps | 63 |
| prefill_tokens | 2048 |
| decode_tokens | 252 |
| prefill_time_s | 0.1487 |
| decode_time_s | 1.7378 |
| elapsed_s | 3.1805 |
| trace_output | results/rtx5090_qwen3_4b/profile_rope_indexing_cleanup.json |
| profile_memory | True |
| record_shapes | True |
| with_stack | False |

## Operator-Level CUDA Attribution

| category | cuda_total_ms | cpu_total_ms | calls |
| --- | --- | --- | --- |
| Linear/GEMM | 1265.1588 | 711.6495 | 27840 |
| Attention | 5.5890 | 16.3491 | 72 |
| Sampling | 4.2945 | 3.5991 | 192 |
| Other | 1.9027 | 775.3503 | 87969 |

Operator attribution is useful for understanding which model components cause CUDA work. It may include child CUDA kernels, so do not sum it as wall-clock time.

## CUDA Kernel Self-Time Categories

| category | self_cuda_time_ms | self_cpu_time_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| Linear/GEMM | 421.3444 | 0.0000 | 11548 | 36.4864 |
| Attention | 64.5959 | 0.0000 | 2304 | 28.0364 |
| Kernel launch/runtime | 17.9592 | 101.7603 | 23584 | 0.7615 |
| Activation | 14.0081 | 0.0000 | 2304 | 6.0799 |
| Normalization | 13.5198 | 0.0000 | 9280 | 1.4569 |
| RoPE | 6.5025 | 0.0000 | 4608 | 1.4111 |
| KV cache store | 2.5525 | 0.0000 | 2304 | 1.1079 |
| Sampling | 2.2931 | 0.0000 | 128 | 17.9145 |
| Other | 0.9427 | 0.0000 | 386 | 2.4423 |

Kernel category self-time sum: 543.7181 ms. Profiler self CUDA total: 525.9020 ms.

Kernel self-time categories are better for deciding low-level optimization targets. Profiler overhead and CUDA asynchronous execution mean these numbers should explain bottleneck shape, not replace wall-clock E2E latency.

## Attention Kernel Events

| name | self_cuda_time_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- |
| _triton_paged_decode_v2_kernel | 61.8014 | 2268 | 27.2493 |
| void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, true, false, false, false, true, false, false>(flash::Flash_fwd_params) | 2.7945 | 36 | 77.6246 |

## Index / Gather Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| aten::index_select | 0.1804 | 230.1590 | 64 | 2.8195 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 0.0076 | 0.0000 | 2 | 3.8240 |
| aten::gather | 0.0064 | 0.1873 | 1 | 6.4000 |
| aten::index | 0.0012 | 0.1194 | 1 | 1.2480 |

## Kernel Launch Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| cuLaunchKernelEx | 17.9204 | 70.8985 | 20764 | 0.8631 |
| cudaLaunchKernelExC | 0.0339 | 12.4926 | 2268 | 0.0150 |
| cudaLaunchKernel | 0.0029 | 18.4860 | 551 | 0.0052 |

## Allocation / Copy Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| aten::copy_ | 0.3168 | 25.3967 | 513 | 0.6174 |
| aten::empty_like | 0.0000 | 75.0376 | 16256 | 0.0000 |
| aten::empty_strided | 0.0000 | 51.8958 | 16769 | 0.0000 |
| aten::empty | 0.0000 | 14.5495 | 2860 | 0.0000 |

## Top Ops

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                  nano_vllm_engine_step         0.00%       0.000us         0.00%       0.000us       0.000us        1.858s       353.32%        1.858s      29.033ms           0 B           0 B           0 B           0 B            64
                                  nano_vllm_engine_step        60.52%        1.140s       100.00%        1.883s      29.425ms       0.000us         0.00%     446.653ms       6.979ms           0 B      -2.00 KB           0 B      -9.76 GB            64
                                           aten::linear         0.99%      18.641ms        14.61%     275.147ms      29.649us       0.000us         0.00%     421.720ms      45.444us           0 B           0 B       4.81 GB           0 B          9280
                                           aten::matmul         0.63%      11.783ms        11.90%     224.143ms      24.153us       0.000us         0.00%     421.720ms      45.444us           0 B           0 B       4.81 GB           0 B          9280
                                               aten::mm         8.63%     162.465ms        11.28%     212.360ms      22.884us     421.344ms        80.12%     421.720ms      45.444us           0 B           0 B       4.81 GB       4.81 GB          9280
void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...         0.00%       0.000us         0.00%       0.000us       0.000us     351.311ms        66.80%     351.311ms      38.453us           0 B           0 B           0 B           0 B          9136
                         _triton_paged_decode_v2_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      61.801ms        11.75%      61.801ms      27.249us           0 B           0 B           0 B           0 B          2268
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      33.668ms         6.40%      33.668ms     311.742us           0 B           0 B           0 B           0 B           108
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      33.533ms         6.38%      33.533ms     931.481us           0 B           0 B           0 B           0 B            36
                                       cuLaunchKernelEx         3.76%      70.899ms         3.76%      70.899ms       3.414us      17.920ms         3.41%      17.920ms       0.863us           0 B           0 B           0 B           0 B         20764
                                   _silu_and_mul_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      14.008ms         2.66%      14.008ms       6.080us           0 B           0 B           0 B           0 B          2304
                                   _add_rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       7.336ms         1.39%       7.336ms       1.592us           0 B           0 B           0 B           0 B          4608
                               _rotary_embedding_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.502ms         1.24%       6.502ms       1.411us           0 B           0 B           0 B           0 B          4608
                                       _rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.184ms         1.18%       6.184ms       1.324us           0 B           0 B           0 B           0 B          4672
void cublasLt::splitKreduce_kernel<32, 16, int, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us       2.832ms         0.54%       2.832ms       1.249us           0 B           0 B           0 B           0 B          2268
                                    FlashAttnVarlenFunc         0.23%       4.354ms         0.55%      10.403ms     288.963us       0.000us         0.00%       2.794ms      77.625us           0 B           0 B     576.00 MB      -9.02 MB            36
                 flash_attn::_flash_attn_varlen_forward         0.18%       3.418ms         0.32%       5.946ms     165.178us       2.794ms         0.53%       2.794ms      77.625us           0 B           0 B     585.02 MB           0 B            36
void flash::flash_fwd_kernel<Flash_fwd_kernel_traits...         0.00%       0.000us         0.00%       0.000us       0.000us       2.794ms         0.53%       2.794ms      77.625us           0 B           0 B           0 B           0 B            36
                                   store_kvcache_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       2.553ms         0.49%       2.553ms       1.108us           0 B           0 B           0 B           0 B          2304
                                          aten::softmax         0.01%     134.554us         0.05%     938.567us      14.665us       0.000us         0.00%       1.984ms      31.003us           0 B           0 B     148.38 MB           0 B            64
                                         aten::_softmax         0.03%     522.052us         0.04%     804.013us      12.563us       1.984ms         0.38%       1.984ms      31.003us           0 B           0 B     148.38 MB     148.38 MB            64
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       1.984ms         0.38%       1.984ms      31.003us           0 B           0 B           0 B           0 B            64
                                Activity Buffer Request         0.51%       9.604ms         0.57%      10.813ms       1.802ms      23.968us         0.00%       1.341ms     223.505us           0 B           0 B      80.00 KB      -3.52 MB             6
                                  cudaStreamIsCapturing         0.02%     299.994us         0.02%     299.994us       0.357us     387.450us         0.07%     387.450us       0.461us           0 B           0 B           0 B           0 B           841
                               cudaEventRecordWithFlags         0.02%     308.504us         0.02%     308.504us       0.803us     385.853us         0.07%     385.853us       1.005us           0 B           0 B           0 B           0 B           384
                                  Lazy Function Loading         0.02%     445.060us         0.02%     445.060us      89.012us     354.107us         0.07%     354.107us      70.821us           0 B           0 B           0 B           0 B             5
                                           aten::argmax         0.05%       1.030ms         0.10%       1.856ms      29.008us     326.142us         0.06%     326.142us       5.096us           0 B           0 B      32.00 KB      32.00 KB            64
                                               aten::to         0.05%       1.012ms         1.59%      29.986ms      33.430us       0.000us         0.00%     316.751us       0.353us       2.00 KB           0 B     148.60 MB           0 B           897
                                         aten::_to_copy         0.10%       1.817ms         1.54%      28.974ms      56.480us       0.000us         0.00%     316.751us       0.617us       2.00 KB           0 B     148.60 MB           0 B           513
                                            aten::copy_         0.14%       2.701ms         1.35%      25.397ms      49.506us     316.751us         0.06%     316.751us       0.617us           0 B           0 B           0 B           0 B           513
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.883s
Self CUDA time total: 525.902ms

```
