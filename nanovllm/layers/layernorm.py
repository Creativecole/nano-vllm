import torch
from torch import nn
import triton
import triton.language as tl


def _next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    return n + 1


@triton.jit
def _rms_norm_kernel(
    x_ptr,          # [N, D]
    out_ptr,        # [N, D]
    w_ptr,          # [D]
    stride_x,       # stride of x along dim 0
    stride_out,     # stride of out along dim 0
    D,              # actual hidden size (runtime value)
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # load one full row — single HBM read
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # variance + rsqrt
    var = tl.sum(x * x, axis=0) / D
    rrms = tl.rsqrt(var + eps)

    # normalize and scale in float32, then cast — single HBM write
    out = (x * rrms * w).to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


@triton.jit
def _add_rms_norm_kernel(
    x_ptr,          # [N, D]  input (attn / mlp output)
    residual_ptr,   # [N, D]  residual stream (read)
    out_ptr,        # [N, D]  normalized output
    new_res_ptr,    # [N, D]  new residual (= x + residual)
    w_ptr,          # [D]
    stride_x,
    stride_res,
    stride_out,
    stride_new_res,
    D,              # actual hidden size (runtime value)
    BLOCK_D: tl.constexpr,
    eps: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # load x and residual — two HBM reads
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + row * stride_res + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # fused add
    x = x + res

    # store new residual — one HBM write
    tl.store(new_res_ptr + row * stride_new_res + cols, x.to(new_res_ptr.dtype.element_ty), mask=mask)

    # rms norm
    var = tl.sum(x * x, axis=0) / D
    rrms = tl.rsqrt(var + eps)

    out = (x * rrms * w).to(out_ptr.dtype.element_ty)
    # store normalized output — one HBM write
    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.hidden_size = hidden_size
        self.block_d = _next_power_of_2(hidden_size)

    def _rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        N = x.shape[:-1].numel()
        _rms_norm_kernel[(N,)](
            x, out, self.weight,
            x.stride(-2), out.stride(-2),
            self.hidden_size,
            BLOCK_D=self.block_d,
            eps=self.eps,
        )
        return out

    def _add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(x)
        new_residual = torch.empty_like(x)
        N = x.shape[:-1].numel()
        _add_rms_norm_kernel[(N,)](
            x, residual, out, new_residual, self.weight,
            x.stride(-2), residual.stride(-2), out.stride(-2), new_residual.stride(-2),
            self.hidden_size,
            BLOCK_D=self.block_d,
            eps=self.eps,
        )
        return out, new_residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._rms_forward(x)
        else:
            return self._add_rms_forward(x, residual)
