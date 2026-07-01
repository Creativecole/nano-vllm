# nano-vLLM E2E PyTorch Profiler

| Metric | Value |
|---|---:|
| model | ../models/Qwen3-4B |
| gpu | NVIDIA GeForce RTX 5090 |
| attn_backend | triton_paged_decode |
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
| prefill_time_s | 0.0806 |
| decode_time_s | 1.5787 |
| elapsed_s | 2.8610 |
| trace_output | results/rtx5090_qwen3_4b/profile_paged_decode_v1.json |
| profile_memory | True |
| record_shapes | True |
| with_stack | False |

## Operator-Level CUDA Attribution

| category | cuda_total_ms | cpu_total_ms | calls |
| --- | --- | --- | --- |
| Linear/GEMM | 1263.6877 | 617.2268 | 27840 |
| Attention | 5.5753 | 4.7742 | 72 |
| Sampling | 4.2908 | 3.4694 | 192 |
| Other | 1.9025 | 651.5508 | 87969 |

Operator attribution is useful for understanding which model components cause CUDA work. It may include child CUDA kernels, so do not sum it as wall-clock time.

## CUDA Kernel Self-Time Categories

| category | self_cuda_time_ms | self_cpu_time_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| Linear/GEMM | 420.8650 | 0.0000 | 11548 | 36.4448 |
| Attention | 83.9849 | 0.0000 | 2304 | 36.4518 |
| Kernel launch/runtime | 17.9591 | 94.8178 | 23584 | 0.7615 |
| Activation | 13.9798 | 0.0000 | 2304 | 6.0676 |
| Normalization | 13.5063 | 0.0000 | 9280 | 1.4554 |
| RoPE | 6.5041 | 0.0000 | 4608 | 1.4115 |
| KV cache store | 2.5510 | 0.0000 | 2304 | 1.1072 |
| Sampling | 2.2913 | 0.0000 | 128 | 17.9006 |
| Other | 0.9430 | 0.0000 | 386 | 2.4431 |

Kernel category self-time sum: 562.5846 ms. Profiler self CUDA total: 544.7680 ms.

Kernel self-time categories are better for deciding low-level optimization targets. Profiler overhead and CUDA asynchronous execution mean these numbers should explain bottleneck shape, not replace wall-clock E2E latency.

## Attention Kernel Events

| name | self_cuda_time_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- |
| _triton_paged_decode_kernel | 81.1973 | 2268 | 35.8013 |
| void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, true, false, false, false, true, false, false>(flash::Flash_fwd_params) | 2.7877 | 36 | 77.4348 |

## Index / Gather Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| aten::index_select | 0.1812 | 102.3327 | 64 | 2.8308 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 0.0069 | 0.0000 | 2 | 3.4715 |
| aten::gather | 0.0057 | 0.0541 | 1 | 5.6630 |
| aten::index | 0.0013 | 0.0728 | 1 | 1.2800 |

## Kernel Launch Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| cuLaunchKernelEx | 17.9205 | 69.5096 | 20764 | 0.8631 |
| cudaLaunchKernelExC | 0.0335 | 12.7422 | 2268 | 0.0148 |
| cudaLaunchKernel | 0.0030 | 12.6743 | 551 | 0.0054 |

## Allocation / Copy Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| aten::copy_ | 0.3166 | 77.2012 | 513 | 0.6172 |
| aten::empty_like | 0.0000 | 68.6322 | 16256 | 0.0000 |
| aten::empty_strided | 0.0000 | 47.4874 | 16769 | 0.0000 |
| aten::empty | 0.0000 | 8.6110 | 2860 | 0.0000 |

## Top Ops

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                  nano_vllm_engine_step         0.00%       0.000us         0.00%       0.000us       0.000us        1.650s       302.79%        1.650s      25.774ms           0 B           0 B           0 B           0 B            64
                                  nano_vllm_engine_step        63.16%        1.047s       100.00%        1.658s      25.905ms       0.000us         0.00%     446.091ms       6.970ms           0 B      -2.00 KB           0 B      -9.76 GB            64
                                           aten::linear         1.02%      16.946ms        14.55%     241.194ms      25.991us       0.000us         0.00%     421.229ms      45.391us           0 B           0 B       4.81 GB           0 B          9280
                                           aten::matmul         0.67%      11.160ms        11.68%     193.597ms      20.862us       0.000us         0.00%     421.229ms      45.391us           0 B           0 B       4.81 GB           0 B          9280
                                               aten::mm         8.03%     133.193ms        11.00%     182.436ms      19.659us     420.865ms        77.26%     421.229ms      45.391us           0 B           0 B       4.81 GB       4.81 GB          9280
void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...         0.00%       0.000us         0.00%       0.000us       0.000us     351.159ms        64.46%     351.159ms      38.437us           0 B           0 B           0 B           0 B          9136
                            _triton_paged_decode_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      81.197ms        14.90%      81.197ms      35.801us           0 B           0 B           0 B           0 B          2268
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      33.507ms         6.15%      33.507ms     310.245us           0 B           0 B           0 B           0 B           108
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      33.373ms         6.13%      33.373ms     927.030us           0 B           0 B           0 B           0 B            36
                                       cuLaunchKernelEx         4.19%      69.510ms         4.19%      69.510ms       3.348us      17.921ms         3.29%      17.921ms       0.863us           0 B           0 B           0 B           0 B         20764
                                   _silu_and_mul_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      13.980ms         2.57%      13.980ms       6.068us           0 B           0 B           0 B           0 B          2304
                                   _add_rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       7.328ms         1.35%       7.328ms       1.590us           0 B           0 B           0 B           0 B          4608
                               _rotary_embedding_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.504ms         1.19%       6.504ms       1.411us           0 B           0 B           0 B           0 B          4608
                                       _rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.179ms         1.13%       6.179ms       1.322us           0 B           0 B           0 B           0 B          4672
void cublasLt::splitKreduce_kernel<32, 16, int, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us       2.826ms         0.52%       2.826ms       1.246us           0 B           0 B           0 B           0 B          2268
                                    FlashAttnVarlenFunc         0.07%       1.239ms         0.18%       3.021ms      83.923us       0.000us         0.00%       2.788ms      77.435us           0 B           0 B     576.00 MB      -9.02 MB            36
                 flash_attn::_flash_attn_varlen_forward         0.06%       1.042ms         0.11%       1.753ms      48.693us       2.788ms         0.51%       2.788ms      77.435us           0 B           0 B     585.02 MB           0 B            36
void flash::flash_fwd_kernel<Flash_fwd_kernel_traits...         0.00%       0.000us         0.00%       0.000us       0.000us       2.788ms         0.51%       2.788ms      77.435us           0 B           0 B           0 B           0 B            36
                                   store_kvcache_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       2.551ms         0.47%       2.551ms       1.107us           0 B           0 B           0 B           0 B          2304
                                          aten::softmax         0.01%     128.579us         0.06%     930.951us      14.546us       0.000us         0.00%       1.983ms      30.979us           0 B           0 B     148.38 MB           0 B            64
                                         aten::_softmax         0.03%     509.810us         0.05%     802.372us      12.537us       1.983ms         0.36%       1.983ms      30.979us           0 B           0 B     148.38 MB     148.38 MB            64
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       1.983ms         0.36%       1.983ms      30.979us           0 B           0 B           0 B           0 B            64
                                Activity Buffer Request         0.55%       9.069ms         0.62%      10.333ms       1.722ms      22.686us         0.00%       1.366ms     227.697us           0 B           0 B     -20.00 KB      -3.64 MB             6
                                  cudaStreamIsCapturing         0.02%     254.076us         0.02%     254.076us       0.302us     387.263us         0.07%     387.263us       0.460us           0 B           0 B           0 B           0 B           841
                               cudaEventRecordWithFlags         0.02%     286.222us         0.02%     286.222us       0.745us     385.884us         0.07%     385.884us       1.005us           0 B           0 B           0 B           0 B           384
                                  Lazy Function Loading         0.02%     256.926us         0.02%     256.926us      51.385us     344.577us         0.06%     344.577us      68.915us           0 B           0 B           0 B           0 B             5
                                           aten::argmax         0.06%     981.274us         0.10%       1.736ms      27.127us     325.538us         0.06%     325.538us       5.087us           0 B           0 B      32.00 KB      32.00 KB            64
                                               aten::to         0.06%     920.921us         4.90%      81.249ms      90.579us       0.000us         0.00%     316.623us       0.353us       2.00 KB           0 B     148.60 MB           0 B           897
                                         aten::_to_copy         0.10%       1.630ms         4.85%      80.328ms     156.585us       0.000us         0.00%     316.623us       0.617us       2.00 KB           0 B     148.60 MB           0 B           513
                                            aten::copy_         0.15%       2.515ms         4.66%      77.201ms     150.490us     316.623us         0.06%     316.623us       0.617us           0 B           0 B           0 B           0 B           513
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.658s
Self CUDA time total: 544.768ms

```
