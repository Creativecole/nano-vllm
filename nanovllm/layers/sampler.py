import torch
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _top_k_softmax_sample_kernel(
    topk_logits_ptr,    # [N, K]   — top-k logits already selected
    topk_indices_ptr,   # [N, K]   — original vocab indices of top-k
    temperatures_ptr,   # [N]
    output_ptr,         # [N]      — sampled token ids
    noise_ptr,          # [N, K]   — pre-generated Exp(1) noise
    stride_logits,      # stride along batch dim for topk_logits
    stride_indices,     # stride along batch dim for topk_indices
    stride_noise,       # stride along batch dim for noise
    K: tl.constexpr,
):
    """Fused softmax + Gumbel-max sampling over top-k logits."""
    row = tl.program_id(0)
    cols = tl.arange(0, K)

    # load top-k logits and apply temperature
    temp = tl.load(temperatures_ptr + row)
    logits = tl.load(topk_logits_ptr + row * stride_logits + cols).to(tl.float32)
    logits = logits / temp

    # online softmax: subtract max for numerical stability
    max_logit = tl.max(logits, axis=0)
    logits = logits - max_logit
    exp_logits = tl.exp(logits)

    # Gumbel-max trick: argmax(probs / noise) where noise ~ Exp(1)
    # This is equivalent to categorical sampling from softmax(logits)
    noise = tl.load(noise_ptr + row * stride_noise + cols)
    noise = tl.maximum(noise, 1e-10)
    scores = exp_logits / noise

    # argmax within top-k, then look up original vocab index
    local_idx = tl.argmax(scores, axis=0)
    token_id = tl.load(topk_indices_ptr + row * stride_indices + local_idx)
    tl.store(output_ptr + row, token_id)


class Sampler(nn.Module):

    def __init__(self, top_k: int | None = None):
        super().__init__()
        self.top_k = top_k

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        if logits is None:
            return None
        if self.top_k is None:
            logits = logits.float().div_(temperatures.unsqueeze(dim=1))
            probs = torch.softmax(logits, dim=-1)
            return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)

        N, V = logits.shape
        K = min(self.top_k, V)

        # step 1: torch.topk on GPU — efficient cuDNN/cuBLAS kernel, avoids full softmax
        topk_logits, topk_indices = torch.topk(logits.float(), K, dim=-1)

        # step 2: pre-generate exponential noise on GPU
        noise = torch.empty(N, K, dtype=torch.float32, device=logits.device).exponential_(1)

        # step 3: fused softmax + Gumbel-max sampling in Triton
        output = torch.empty(N, dtype=torch.int64, device=logits.device)
        # pad K to next power of 2 for Triton
        K_padded = 1
        while K_padded < K:
            K_padded *= 2

        if K_padded == K:
            _top_k_softmax_sample_kernel[(N,)](
                topk_logits, topk_indices, temperatures, output, noise,
                topk_logits.stride(0), topk_indices.stride(0), noise.stride(0),
                K=K_padded,
            )
        else:
            # need to pad — but K is typically 50, padded to 64
            topk_logits_pad = torch.full((N, K_padded), float('-inf'), dtype=torch.float32, device=logits.device)
            topk_logits_pad[:, :K] = topk_logits
            topk_indices_pad = torch.zeros(N, K_padded, dtype=topk_indices.dtype, device=logits.device)
            topk_indices_pad[:, :K] = topk_indices
            noise_pad = torch.ones(N, K_padded, dtype=torch.float32, device=logits.device)
            noise_pad[:, :K] = noise
            _top_k_softmax_sample_kernel[(N,)](
                topk_logits_pad, topk_indices_pad, temperatures, output, noise_pad,
                topk_logits_pad.stride(0), topk_indices_pad.stride(0), noise_pad.stride(0),
                K=K_padded,
            )
        return output
