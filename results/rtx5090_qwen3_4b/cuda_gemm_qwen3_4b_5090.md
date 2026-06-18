| Shape | M | K | N | torch.matmul | CUDA naive | CUDA tiled | Naive vs torch | Tiled vs torch | Correctness |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| qkv_decode | 1 | 2560 | 6144 | 15.92 us / 1975.4 GFLOP/s | 58.01 us / 542.3 GFLOP/s | 75.85 us / 414.7 GFLOP/s | 0.27x | 0.21x | pass max=0.000412, mean=3.31e-05 |
| o_proj_decode | 1 | 4096 | 2560 | 13.47 us / 1557.1 GFLOP/s | 92.17 us / 227.5 GFLOP/s | 56.00 us / 374.5 GFLOP/s | 0.15x | 0.24x | pass max=0.000389, mean=5.31e-05 |
| gate_up_decode | 1 | 2560 | 19456 | 123.28 us / 808.1 GFLOP/s | 250.04 us / 398.4 GFLOP/s | 198.43 us / 502.0 GFLOP/s | 0.49x | 0.62x | pass max=0.000336, mean=3.01e-05 |
| down_proj_decode | 1 | 9728 | 2560 | 34.00 us / 1465.0 GFLOP/s | 395.85 us / 125.8 GFLOP/s | 182.27 us / 273.3 GFLOP/s | 0.09x | 0.19x | pass max=0.00111, mean=0.000136 |
| qkv_small_batch | 16 | 2560 | 6144 | 38.36 us / 13122.1 GFLOP/s | 116.66 us / 4314.4 GFLOP/s | 76.79 us / 6554.5 GFLOP/s | 0.33x | 0.50x | pass max=0.000381, mean=3.37e-05 |
| o_proj_small_batch | 16 | 4096 | 2560 | 28.32 us / 11849.3 GFLOP/s | 115.85 us / 2896.5 GFLOP/s | 56.53 us / 5935.3 GFLOP/s | 0.24x | 0.50x | pass max=0.000626, mean=5.33e-05 |
| gate_up_small_batch | 16 | 2560 | 19456 | 126.04 us / 12645.5 GFLOP/s | 321.77 us / 4953.3 GFLOP/s | 202.12 us / 7885.8 GFLOP/s | 0.39x | 0.62x | pass max=0.000504, mean=2.99e-05 |
| down_proj_small_batch | 16 | 9728 | 2560 | 51.18 us / 15571.3 GFLOP/s | 476.28 us / 1673.2 GFLOP/s | 195.66 us / 4073.1 GFLOP/s | 0.11x | 0.26x | pass max=0.00168, mean=0.000125 |
| qkv_prefill | 256 | 2560 | 6144 | 141.03 us / 57100.7 GFLOP/s | 1126.43 us / 7149.2 GFLOP/s | 817.26 us / 9853.8 GFLOP/s | 0.13x | 0.17x | pass max=0.000565, mean=3.4e-05 |
| gate_up_prefill | 256 | 2560 | 19456 | 438.92 us / 58099.8 GFLOP/s | 4209.07 us / 6058.7 GFLOP/s | 2688.49 us / 9485.4 GFLOP/s | 0.10x | 0.16x | pass max=0.000565, mean=2.98e-05 |
| down_proj_prefill | 256 | 9728 | 2560 | 258.26 us / 49372.3 GFLOP/s | 2008.55 us / 6348.2 GFLOP/s | 1378.65 us / 9248.7 GFLOP/s | 0.13x | 0.19x | pass max=0.0025, mean=0.000127 |
