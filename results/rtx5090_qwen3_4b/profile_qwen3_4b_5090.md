# nano-vLLM E2E PyTorch Profiler

| Metric | Value |
|---|---:|
| model | ../models/Qwen3-4B |
| gpu | NVIDIA GeForce RTX 5090 |
| prompt_len | 512 |
| num_prompts | 4 |
| max_tokens | 128 |
| profiled_steps | 64 |
| prefill_steps | 1 |
| decode_steps | 63 |
| prefill_tokens | 2048 |
| decode_tokens | 252 |
| elapsed_s | 4.1670 |
| trace_output | results/rtx5090_qwen3_4b/profile_qwen3_4b_5090.json |
| profile_memory | True |
| record_shapes | True |
| with_stack | False |

## Top Ops

```text
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg       CPU Mem  Self CPU Mem      CUDA Mem  Self CUDA Mem    # of Calls  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                                  nano_vllm_engine_step         0.00%       0.000us         0.00%       0.000us       0.000us        2.024s       404.83%        2.024s      31.632ms           0 B           0 B           0 B           0 B            64  
                                  nano_vllm_engine_step        55.00%        1.123s       100.00%        2.043s      31.916ms       0.000us         0.00%     456.100ms       7.127ms           0 B      -2.00 KB           0 B     -10.64 GB            64  
                                           aten::linear         1.20%      24.488ms        12.98%     265.132ms      28.570us       0.000us         0.00%     424.531ms      45.747us           0 B           0 B       4.81 GB           0 B          9280  
                                           aten::matmul         0.73%      14.906ms        10.06%     205.582ms      22.153us       0.000us         0.00%     424.531ms      45.747us           0 B           0 B       4.81 GB           0 B          9280  
                                               aten::mm         6.58%     134.363ms         9.33%     190.676ms      20.547us     424.475ms        84.88%     424.531ms      45.747us           0 B           0 B       4.81 GB       4.81 GB          9280  
void cutlass::Kernel2<cutlass_80_wmma_tensorop_bf16_...         0.00%       0.000us         0.00%       0.000us       0.000us     353.075ms        70.61%     353.075ms      38.647us           0 B           0 B           0 B           0 B          9136  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      34.279ms         6.85%      34.279ms     317.397us           0 B           0 B           0 B           0 B           108  
void cutlass::Kernel2<cutlass_80_tensorop_bf16_s1681...         0.00%       0.000us         0.00%       0.000us       0.000us      34.201ms         6.84%      34.201ms     950.033us           0 B           0 B           0 B           0 B            36  
void flash::flash_fwd_splitkv_kernel<Flash_fwd_kerne...         0.00%       0.000us         0.00%       0.000us       0.000us      21.105ms         4.22%      21.105ms       9.305us           0 B           0 B           0 B           0 B          2268  
                                   _silu_and_mul_kernel         0.00%       0.000us         0.00%       0.000us       0.000us      14.493ms         2.90%      14.493ms       6.290us           0 B           0 B           0 B           0 B          2304  
                                       cuLaunchKernelEx         3.81%      77.881ms         3.90%      79.655ms       4.307us      11.640ms         2.33%      11.640ms       0.629us           0 B           0 B           0 B           0 B         18496  
                                   _add_rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       7.536ms         1.51%       7.536ms       1.635us           0 B           0 B           0 B           0 B          4608  
void flash::flash_fwd_splitkv_combine_kernel<Flash_f...         0.00%       0.000us         0.00%       0.000us       0.000us       6.538ms         1.31%       6.538ms       2.883us           0 B           0 B           0 B           0 B          2268  
                                       _rms_norm_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       6.371ms         1.27%       6.371ms       1.364us           0 B           0 B           0 B           0 B          4672  
void at::native::vectorized_gather_kernel<16, long>(...         0.00%       0.000us         0.00%       0.000us       0.000us       5.526ms         1.11%       5.526ms       1.199us           0 B           0 B           0 B           0 B          4610  
                                            aten::index         2.84%      57.999ms         4.77%      97.412ms      21.135us       5.521ms         1.10%       5.526ms       1.199us           0 B           0 B      40.45 MB      40.45 MB          4609  
                               _rotary_embedding_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       5.035ms         1.01%       5.035ms       1.093us           0 B           0 B           0 B           0 B          4608  
                                 cudaDeviceGetAttribute         0.27%       5.504ms         0.27%       5.504ms       0.339us       4.061ms         0.81%       4.061ms       0.250us           0 B           0 B           0 B           0 B         16228  
void cublasLt::splitKreduce_kernel<32, 16, int, __nv...         0.00%       0.000us         0.00%       0.000us       0.000us       2.920ms         0.58%       2.920ms       1.287us           0 B           0 B           0 B           0 B          2268  
                                    FlashAttnVarlenFunc         0.23%       4.652ms         0.54%      11.065ms     307.373us       0.000us         0.00%       2.858ms      79.380us           0 B           0 B     576.00 MB      -9.02 MB            36  
                 flash_attn::_flash_attn_varlen_forward         0.19%       3.813ms         0.31%       6.292ms     174.766us       2.858ms         0.57%       2.858ms      79.380us           0 B           0 B     585.02 MB           0 B            36  
void flash::flash_fwd_kernel<Flash_fwd_kernel_traits...         0.00%       0.000us         0.00%       0.000us       0.000us       2.858ms         0.57%       2.858ms      79.380us           0 B           0 B           0 B           0 B            36  
                                   store_kvcache_kernel         0.00%       0.000us         0.00%       0.000us       0.000us       2.672ms         0.53%       2.672ms       1.160us           0 B           0 B           0 B           0 B          2304  
                                       cudaLaunchKernel         3.01%      61.488ms         3.09%      63.063ms       6.505us       2.432ms         0.49%       2.435ms       0.251us           0 B           0 B           0 B           0 B          9695  
                                          aten::softmax         0.01%     154.860us         0.06%       1.243ms      19.421us       0.000us         0.00%       2.024ms      31.626us           0 B           0 B     148.38 MB           0 B            64  
                                         aten::_softmax         0.03%     708.337us         0.05%       1.088ms      17.001us       2.024ms         0.40%       2.024ms      31.626us           0 B           0 B     148.38 MB     148.38 MB            64  
void at::native::(anonymous namespace)::cunn_SoftMax...         0.00%       0.000us         0.00%       0.000us       0.000us       2.024ms         0.40%       2.024ms      31.626us           0 B           0 B           0 B           0 B            64  
                                   cudaFuncSetAttribute         0.29%       5.941ms         1.23%      25.138ms      10.911us       1.213ms         0.24%       1.213ms       0.526us           0 B           0 B           0 B           0 B          2304  
                                Activity Buffer Request         0.55%      11.276ms         0.64%      13.059ms       1.866ms      23.776us         0.00%     971.328us     138.761us           0 B           0 B      80.00 KB      -5.02 MB             7  
                                           aten::argmax         0.06%       1.206ms         0.11%       2.206ms      34.464us     329.888us         0.07%     329.888us       5.155us           0 B           0 B      32.00 KB      32.00 KB            64  
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 2.043s
Self CUDA time total: 500.061ms

```
