import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import triton
import triton.language as tl


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


# ============================================================
# Triton tiled GEMV kernel for small batch sizes (bs <= 16)
# ============================================================
# Decode often runs with tiny M, where launching a full GEMM can be
# inefficient. This kernel computes one batch row and BLOCK_N output
# columns per program, accumulating the K dimension inside that program.

@triton.jit
def _gemv_kernel(
    x_ptr,          # [M, K]
    w_ptr,          # [N, K]
    out_ptr,        # [M, N]
    M,              # batch size
    N,              # output features
    K,              # input features
    stride_x_m,
    stride_w_n,
    stride_out_m,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Grid: (cdiv(N, BLOCK_N), M)
    Each program computes a [1, BLOCK_N] tile of the output by
    accumulating over K in chunks of BLOCK_K.
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    n_offset = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = n_offset < N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offset = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offset < K

        # load x[pid_m, k_start:k_start+BLOCK_K]
        x = tl.load(x_ptr + pid_m * stride_x_m + k_offset, mask=k_mask, other=0.0).to(tl.float32)
        # load w[n_offset, k_start:k_start+BLOCK_K] — need 2D load
        # w is [N, K], row-major: w[n, k] = w_ptr + n * stride_w_n + k
        w = tl.load(
            w_ptr + n_offset[:, None] * stride_w_n + k_offset[None, :],
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)  # [BLOCK_N, BLOCK_K]

        # dot product: acc += sum(w * x, axis=1)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(out_ptr + pid_m * stride_out_m + n_offset, acc.to(out_ptr.dtype.element_ty), mask=n_mask)


def triton_gemv(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """GEMV for small batch: x @ weight.T + bias"""
    M = x.shape[0]
    N, K = weight.shape
    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    # tuning parameters
    BLOCK_N = 128
    BLOCK_K = 128

    grid = (triton.cdiv(N, BLOCK_N), M)
    _gemv_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), weight.stride(0), out.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    if bias is not None:
        out += bias
    return out


# RTX 5090 benchmark showed this educational Triton GEMV is slower than
# cuBLAS for Qwen3 decode shapes, so keep it available for experiments but
# do not route production forwards through it by default.
GEMV_THRESHOLD = 0


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= GEMV_THRESHOLD:
            return triton_gemv(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= GEMV_THRESHOLD:
            return triton_gemv(x, self.weight, self.bias)
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] <= GEMV_THRESHOLD:
            y = triton_gemv(x, self.weight, self.bias if self.tp_rank == 0 else None)
        else:
            y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
