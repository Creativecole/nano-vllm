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
def _silu_and_mul_kernel(
    x_ptr,          # [N, 2*D]  — gate | up packed along last dim
    out_ptr,        # [N, D]
    stride_x,       # stride of x along dim 0
    stride_out,     # stride of out along dim 0
    D,              # actual half-size (runtime value)
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D

    # gate = first half, up = second half
    gate = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(x_ptr + row * stride_x + D + cols, mask=mask, other=0.0).to(tl.float32)

    # silu(gate) * up = gate * sigmoid(gate) * up
    gate_sigmoid = tl.sigmoid(gate)
    out = (gate * gate_sigmoid * up).to(out_ptr.dtype.element_ty)

    tl.store(out_ptr + row * stride_out + cols, out, mask=mask)


class SiluAndMul(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.shape[:-1].numel()
        D = x.shape[-1] // 2
        BLOCK_D = _next_power_of_2(D)
        out = torch.empty(*x.shape[:-1], D, dtype=x.dtype, device=x.device)
        _silu_and_mul_kernel[(N,)](
            x, out,
            x.stride(-2), out.stride(-2),
            D, BLOCK_D=BLOCK_D,
        )
        return out
