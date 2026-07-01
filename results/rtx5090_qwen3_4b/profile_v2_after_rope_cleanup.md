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
| prefill_time_s | 0.0808 |
| decode_time_s | 1.4353 |
| elapsed_s | 2.7180 |
| trace_output | results/rtx5090_qwen3_4b/profile_v2_after_rope_cleanup.json |
| profile_memory | True |
| record_shapes | True |
| with_stack | False |

## Operator-Level CUDA Attribution

| category | cuda_total_ms | cpu_total_ms | calls |
| --- | --- | --- | --- |
| Linear/GEMM | 1264.8341 | 505.9575 | 27840 |
| Attention | 5.5697 | 4.8999 | 72 |
| Sampling | 4.2940 | 3.3551 | 192 |
| Other | 1.9078 | 641.1920 | 87969 |

Operator attribution is useful for understanding which model components cause CUDA work. It may include child CUDA kernels, so do not sum it as wall-clock time.

## CUDA Kernel Self-Time Categories

| category | self_cuda_time_ms | self_cpu_time_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| Linear/GEMM | 421.1892 | 0.0000 | 11548 | 36.4729 |
| Attention | 64.6498 | 0.0000 | 2304 | 28.0598 |
| Kernel launch/runtime | 17.9858 | 88.0494 | 23584 | 0.7626 |
| Activation | 14.0044 | 0.0000 | 2304 | 6.0783 |
| Normalization | 13.5193 | 0.0000 | 9280 | 1.4568 |
| RoPE | 6.4986 | 0.0000 | 4608 | 1.4103 |
| KV cache store | 2.5552 | 0.0000 | 2304 | 1.1090 |
| Sampling | 2.2928 | 0.0000 | 128 | 17.9123 |
| Other | 0.9439 | 0.0000 | 386 | 2.4454 |

Kernel category self-time sum: 543.6390 ms. Profiler self CUDA total: 525.7960 ms.

Kernel self-time categories are better for deciding low-level optimization targets. Profiler overhead and CUDA asynchronous execution mean these numbers should explain bottleneck shape, not replace wall-clock E2E latency.

## Attention Kernel Events

| name | self_cuda_time_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- |
| _triton_paged_decode_v2_kernel | 61.8650 | 2268 | 27.2773 |
| void flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4, false, false, cutlass::bfloat16_t, Flash_kernel_traits<128, 128, 64, 4, cutlass::bfloat16_t> >, false, true, false, false, false, true, false, false>(flash::Flash_fwd_params) | 2.7848 | 36 | 77.3565 |

## Index / Gather Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| aten::index_select | 0.1798 | 102.7607 | 64 | 2.8089 |
| void at::native::vectorized_gather_kernel<16, long>(char*, char*, long*, int, long, long, long, long, bool) | 0.0069 | 0.0000 | 2 | 3.4720 |
| aten::gather | 0.0056 | 0.0561 | 1 | 5.6320 |
| aten::index | 0.0013 | 0.0726 | 1 | 1.3120 |

## Kernel Launch Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| cuLaunchKernelEx | 17.9470 | 66.0660 | 20764 | 0.8643 |
| cudaLaunchKernelExC | 0.0340 | 9.4199 | 2268 | 0.0150 |
| cudaLaunchKernel | 0.0027 | 12.6198 | 551 | 0.0049 |

## Allocation / Copy Ops

| name | self_cuda_time_ms | cpu_total_ms | calls | avg_self_cuda_us |
| --- | --- | --- | --- | --- |
| aten::copy_ | 0.3195 | 76.8518 | 513 | 0.6228 |
| aten::empty_like | 0.0000 | 64.7955 | 16256 | 0.0000 |
| aten::empty_strided | 0.0000 | 44.4586 | 16769 | 0.0000 |
| aten::empty | 0.0000 | 8.2194 | 2860 | 0.0000 |

## Top Ops

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                  nano_vllm_engine_step         0.00%       0.000us         0.00%       0.000us       0.000us        1.506s       286.51%        1.506s      23.539ms           0 B           0 B           0 B           0 B            64
                                  nano_vllm_engine_step        62.81%     951.345ms       100.00%        1.515s      23.667ms       0.000us         0.00%     446.523ms       6.977ms           0 B      -2.00 KB           0 B      -9.77 GB            64
                                           aten::linear         1.11%      16.784ms        13.40%     203.001ms      21.875us       0.000us         0.00%     421.611ms      45.432us           0 B           0 B       4.81 GB           0 B          9280
                                           aten::matmul         0.69%      10.464ms        10.35%     156.710ms      16.887us       0.000us         0.00%     421.611ms      45.432us           0 B           0 B       4.81 GB           0 B          9280
                                               aten::mm         6.79%     102.803ms         9.66%     146.247ms      15.759us     421.189ms        80.10%     421.611ms      45.432us           0 B           0 B       4.81 GB       4.81 GB          9280
void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...         0.00%       0.000us         0.00%       0.000us       0.000us     351.568ms        66.86%     351.568ms      38.482us           0 B           0 B           0 B           0 B          9136
                         _triton_paged_decode_v2_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      61.865ms        11.77%      61.865ms      27.277us           0 B           0 B           0 B           0 B          2268
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      33.462ms         6.36%      33.462ms     309.834us           0 B           0 B           0 B           0 B           108
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      33.327ms         6.34%      33.327ms     925.742us           0 B           0 B           0 B           0 B            36
                                       cuLaunchKernelEx         4.36%      66.066ms         4.36%      66.066ms       3.182us      17.947ms         3.41%      17.947ms       0.864us           0 B           0 B           0 B           0 B         20764
                                   _silu_and_mul_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      14.004ms         2.66%      14.004ms       6.078us           0 B           0 B           0 B           0 B          2304
                                   _add_rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       7.339ms         1.40%       7.339ms       1.593us           0 B           0 B           0 B           0 B          4608
                               _rotary_embedding_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.499ms         1.24%       6.499ms       1.410us           0 B           0 B           0 B           0 B          4608
                                       _rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.180ms         1.18%       6.180ms       1.323us           0 B           0 B           0 B           0 B          4672
void cublasLt::splitKreduce_kernel<32, 16, int, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us       2.833ms         0.54%       2.833ms       1.249us           0 B           0 B           0 B           0 B          2268
                                    FlashAttnVarlenFunc         0.08%       1.253ms         0.20%       3.093ms      85.910us       0.000us         0.00%       2.785ms      77.357us           0 B           0 B     576.00 MB      -9.02 MB            36
                 flash_attn::_flash_attn_varlen_forward         0.07%       1.025ms         0.12%       1.807ms      50.199us       2.785ms         0.53%       2.785ms      77.357us           0 B           0 B     585.02 MB           0 B            36
void flash::flash_fwd_kernel<Flash_fwd_kernel_traits...         0.00%       0.000us         0.00%       0.000us       0.000us       2.785ms         0.53%       2.785ms      77.357us           0 B           0 B           0 B           0 B            36
                                   store_kvcache_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       2.555ms         0.49%       2.555ms       1.109us           0 B           0 B           0 B           0 B          2304
                                          aten::softmax         0.01%     124.914us         0.06%     882.859us      13.795us       0.000us         0.00%       1.984ms      31.003us           0 B           0 B     148.38 MB           0 B            64
                                         aten::_softmax         0.03%     489.824us         0.05%     757.945us      11.843us       1.984ms         0.38%       1.984ms      31.003us           0 B           0 B     148.38 MB     148.38 MB            64
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       1.984ms         0.38%       1.984ms      31.003us           0 B           0 B           0 B           0 B            64
                                Activity Buffer Request         0.60%       9.016ms         0.68%      10.355ms       1.726ms      46.365us         0.01%       1.963ms     327.231us           0 B           0 B       5.58 MB      -1.87 MB             6
                                  cudaStreamIsCapturing         0.02%     272.078us         0.02%     272.078us       0.324us     388.230us         0.07%     388.230us       0.462us           0 B           0 B           0 B           0 B           841
                               cudaEventRecordWithFlags         0.02%     271.447us         0.02%     271.447us       0.707us     386.307us         0.07%     386.307us       1.006us           0 B           0 B           0 B           0 B           384
                                  Lazy Function Loading         0.01%     164.896us         0.01%     164.896us      32.979us     344.136us         0.07%     344.136us      68.827us           0 B           0 B           0 B           0 B             5
                                           aten::argmax         0.06%     953.495us         0.11%       1.714ms      26.786us     325.554us         0.06%     325.554us       5.087us           0 B           0 B      32.00 KB      32.00 KB            64
                                               aten::to         0.06%     853.428us         5.33%      80.703ms      89.970us       0.000us         0.00%     319.497us       0.356us       2.00 KB           0 B     148.60 MB           0 B           897
                                         aten::_to_copy         0.10%       1.546ms         5.27%      79.850ms     155.653us       0.000us         0.00%     319.497us       0.623us       2.00 KB           0 B     148.60 MB           0 B           513
                                            aten::copy_         0.16%       2.434ms         5.07%      76.852ms     149.809us     319.497us         0.06%     319.497us       0.623us           0 B           0 B           0 B           0 B           513
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 1.515s
Self CUDA time total: 525.796ms

```
