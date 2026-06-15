#include <torch/extension.h>

#include <cuda_runtime.h>

namespace {

void check_cuda_error() {
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
}

void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
  TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
  TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
  TORCH_CHECK(a.dtype() == torch::kFloat32, "a must be float32");
  TORCH_CHECK(b.dtype() == torch::kFloat32, "b must be float32");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(b.dim() == 2, "b must be 2D");
  TORCH_CHECK(a.size(1) == b.size(0), "a.shape[1] must equal b.shape[0]");
}

__global__ void matmul_naive_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m || col >= n) {
    return;
  }

  float acc = 0.0f;
  for (int kk = 0; kk < k; ++kk) {
    acc += a[row * k + kk] * b[kk * n + col];
  }
  c[row * n + col] = acc;
}

template <int TILE>
__global__ void matmul_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int m,
    int n,
    int k) {
  __shared__ float a_tile[TILE][TILE];
  __shared__ float b_tile[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;

  for (int base = 0; base < k; base += TILE) {
    int a_col = base + threadIdx.x;
    int b_row = base + threadIdx.y;
    a_tile[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    b_tile[threadIdx.y][threadIdx.x] = (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < TILE; ++kk) {
      acc += a_tile[threadIdx.y][kk] * b_tile[kk][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < m && col < n) {
    c[row * n + col] = acc;
  }
}

torch::Tensor matmul_naive(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);
  auto c = torch::empty({a.size(0), b.size(1)}, a.options());
  dim3 block(16, 16);
  dim3 grid((b.size(1) + block.x - 1) / block.x, (a.size(0) + block.y - 1) / block.y);
  matmul_naive_kernel<<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));
  check_cuda_error();
  return c;
}

torch::Tensor matmul_tiled(torch::Tensor a, torch::Tensor b) {
  check_inputs(a, b);
  auto c = torch::empty({a.size(0), b.size(1)}, a.options());
  constexpr int tile = 16;
  dim3 block(tile, tile);
  dim3 grid((b.size(1) + tile - 1) / tile, (a.size(0) + tile - 1) / tile);
  matmul_tiled_kernel<tile><<<grid, block>>>(
      a.data_ptr<float>(),
      b.data_ptr<float>(),
      c.data_ptr<float>(),
      static_cast<int>(a.size(0)),
      static_cast<int>(b.size(1)),
      static_cast<int>(a.size(1)));
  check_cuda_error();
  return c;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_naive", &matmul_naive, "Naive FP32 CUDA GEMM");
  m.def("matmul_tiled", &matmul_tiled, "Shared-memory tiled FP32 CUDA GEMM");
}
