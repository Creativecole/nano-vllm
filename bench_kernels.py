"""
Benchmark script for individual Triton kernel optimizations.
Run on GPU: python bench_kernels.py

Compares the original torch.compile implementations against
the new Triton kernel implementations for each optimized layer.
"""

import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
import torch.distributed as dist

# Initialize a single-GPU process group for the linear layers
dist.init_process_group("gloo", rank=0, world_size=1)


def bench(fn, label, sub_label, description, min_run_time=2.0):
    """Helper to create a benchmark Timer."""
    return benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def benchmark_layernorm():
    """Benchmark RMSNorm: original torch.compile vs Triton kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: RMSNorm (layernorm.py)")
    print("=" * 70)

    hidden_size = 1024  # Qwen3-0.6B hidden_size
    eps = 1e-6
    device = "cuda"

    # --- Original Implementation ---
    class RMSNormOriginal(torch.nn.Module):
        def __init__(self, hidden_size, eps):
            super().__init__()
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(hidden_size, device=device))

        @torch.compile
        def rms_forward(self, x):
            orig_dtype = x.dtype
            x = x.float()
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x.mul_(torch.rsqrt(var + self.eps))
            x = x.to(orig_dtype).mul_(self.weight)
            return x

        @torch.compile
        def add_rms_forward(self, x, residual):
            orig_dtype = x.dtype
            x = x.float().add_(residual.float())
            residual = x.to(orig_dtype)
            var = x.pow(2).mean(dim=-1, keepdim=True)
            x.mul_(torch.rsqrt(var + self.eps))
            x = x.to(orig_dtype).mul_(self.weight)
            return x, residual

    # --- New Implementation ---
    from nanovllm.layers.layernorm import RMSNorm

    results = []
    for N in [1, 16, 128, 1024, 4096]:
        x = torch.randn(N, hidden_size, device=device, dtype=torch.bfloat16)
        residual = torch.randn(N, hidden_size, device=device, dtype=torch.bfloat16)

        orig = RMSNormOriginal(hidden_size, eps).to(device)
        new = RMSNorm(hidden_size, eps).to(device)

        # warmup
        for _ in range(3):
            orig.rms_forward(x.clone())
            new(x.clone())

        # rms_forward
        r1 = bench(lambda: orig.rms_forward(x.clone()), "RMSNorm", f"N={N}", "torch.compile")
        r2 = bench(lambda: new(x.clone()), "RMSNorm", f"N={N}", "Triton")
        results.extend([r1, r2])

        # warmup add_rms
        for _ in range(3):
            orig.add_rms_forward(x.clone(), residual.clone())
            new(x.clone(), residual.clone())

        r3 = bench(lambda: orig.add_rms_forward(x.clone(), residual.clone()), "AddRMSNorm", f"N={N}", "torch.compile")
        r4 = bench(lambda: new(x.clone(), residual.clone()), "AddRMSNorm", f"N={N}", "Triton")
        results.extend([r3, r4])

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_activation():
    """Benchmark SiluAndMul: original torch.compile vs Triton kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: SiluAndMul (activation.py)")
    print("=" * 70)

    device = "cuda"
    intermediate_size = 2816  # Qwen3-0.6B intermediate_size (NOT power of 2!)

    # --- Original ---
    class SiluAndMulOriginal(torch.nn.Module):
        @torch.compile
        def forward(self, x):
            x, y = x.chunk(2, -1)
            return F.silu(x) * y

    # --- New ---
    from nanovllm.layers.activation import SiluAndMul

    results = []
    for N in [1, 16, 128, 1024, 4096]:
        x = torch.randn(N, intermediate_size * 2, device=device, dtype=torch.bfloat16)

        orig = SiluAndMulOriginal()
        new = SiluAndMul()

        # warmup
        for _ in range(3):
            orig(x.clone())
            new(x.clone())

        r1 = bench(lambda: orig(x.clone()), "SiluAndMul", f"N={N}", "torch.compile")
        r2 = bench(lambda: new(x.clone()), "SiluAndMul", f"N={N}", "Triton")
        results.extend([r1, r2])

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_rotary():
    """Benchmark RotaryEmbedding: original torch.compile vs Triton kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: RotaryEmbedding (rotary_embedding.py)")
    print("=" * 70)

    device = "cuda"
    head_dim = 128
    num_heads = 16
    num_kv_heads = 8
    max_pos = 4096

    # --- Original ---
    def apply_rotary_emb_orig(x, cos, sin):
        x1, x2 = torch.chunk(x.float(), 2, dim=-1)
        y1 = x1 * cos - x2 * sin
        y2 = x2 * cos + x1 * sin
        return torch.cat((y1, y2), dim=-1).to(x.dtype)

    class RotaryEmbeddingOriginal(torch.nn.Module):
        def __init__(self):
            super().__init__()
            inv_freq = 1.0 / (10000.0**(torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
            t = torch.arange(max_pos, dtype=torch.float)
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
            self.register_buffer("cos_sin_cache", cache, persistent=False)

        @torch.compile
        def forward(self, positions, query, key):
            cos_sin = self.cos_sin_cache[positions]
            cos, sin = cos_sin.chunk(2, dim=-1)
            query = apply_rotary_emb_orig(query, cos, sin)
            key = apply_rotary_emb_orig(key, cos, sin)
            return query, key

    # --- New ---
    from nanovllm.layers.rotary_embedding import RotaryEmbedding

    results = []
    for N in [1, 16, 128, 1024, 4096]:
        q = torch.randn(N, num_heads, head_dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(N, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        positions = torch.randint(0, max_pos, (N,), device=device, dtype=torch.int64)

        orig = RotaryEmbeddingOriginal().to(device)
        new = RotaryEmbedding(head_dim, head_dim, max_pos, 10000.0).to(device)

        # warmup
        for _ in range(3):
            orig(positions, q.clone(), k.clone())
            new(positions, q.clone(), k.clone())

        r1 = bench(lambda: orig(positions, q.clone(), k.clone()), "RoPE", f"N={N}", "torch.compile")
        r2 = bench(lambda: new(positions, q.clone(), k.clone()), "RoPE", f"N={N}", "Triton")
        results.extend([r1, r2])

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_kvcache():
    """Benchmark store_kvcache: original 1D grid vs new 2D grid."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: store_kvcache (attention.py)")
    print("=" * 70)

    import triton
    import triton.language as tl
    device = "cuda"
    num_kv_heads = 8
    head_dim = 128
    num_blocks = 256
    block_size = 256
    D = num_kv_heads * head_dim

    # --- Original 1D kernel ---
    @triton.jit
    def store_kvcache_kernel_old(
        key_ptr, key_stride, value_ptr, value_stride,
        k_cache_ptr, v_cache_ptr, slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)

    def store_kvcache_old(key, value, k_cache, v_cache, slot_mapping):
        N = key.shape[0]
        D = num_kv_heads * head_dim
        store_kvcache_kernel_old[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)

    # --- New 2D kernel ---
    from nanovllm.layers.attention import store_kvcache as store_kvcache_new

    results = []
    for N in [1, 16, 128, 512, 2048]:
        key = torch.randn(N, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        value = torch.randn(N, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        slot_mapping = torch.randint(0, num_blocks * block_size, (N,), device=device, dtype=torch.int32)

        # warmup
        for _ in range(3):
            store_kvcache_old(key, value, k_cache, v_cache, slot_mapping)
            store_kvcache_new(key, value, k_cache, v_cache, slot_mapping)

        r1 = bench(lambda: store_kvcache_old(key, value, k_cache, v_cache, slot_mapping), "KVCache", f"N={N}", "1D-grid")
        r2 = bench(lambda: store_kvcache_new(key, value, k_cache, v_cache, slot_mapping), "KVCache", f"N={N}", "2D-grid")
        results.extend([r1, r2])

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_sampler():
    """Benchmark Sampler: original full-softmax vs top-k fused."""
    print("\n" + "=" * 70)
    print("BENCHMARK 5: Sampler (sampler.py)")
    print("=" * 70)

    device = "cuda"

    # --- Original ---
    class SamplerOriginal(torch.nn.Module):
        @torch.compile
        def forward(self, logits, temperatures):
            logits = logits.float().div_(temperatures.unsqueeze(dim=1))
            probs = torch.softmax(logits, dim=-1)
            sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            return sample_tokens

    # --- New top-k mode (intentional approximation, not the default sampler path) ---
    from nanovllm.layers.sampler import Sampler

    results = []
    for V in [32000, 151936]:  # LLaMA vocab, Qwen3 vocab
        for N in [1, 16, 64, 256]:
            logits = torch.randn(N, V, device=device, dtype=torch.bfloat16)
            temps = torch.ones(N, device=device, dtype=torch.float32)

            orig = SamplerOriginal()
            new = Sampler(top_k=50)

            # warmup
            for _ in range(3):
                orig(logits.clone(), temps)
                new(logits.clone(), temps)

            r1 = bench(lambda: orig(logits.clone(), temps), "Sampler", f"V={V},N={N}", "torch.compile")
            r2 = bench(lambda: new(logits.clone(), temps), "Sampler", f"V={V},N={N}", "Triton top-k")
            results.extend([r1, r2])

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_linear():
    """Benchmark Linear: F.linear vs Triton GEMV for small batches."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: Linear GEMV (linear.py)")
    print("=" * 70)

    device = "cuda"
    from nanovllm.layers.linear import triton_gemv

    results = []
    # typical Qwen3-0.6B layer dims
    configs = [
        ("QKV", 1024, 1280),       # hidden -> (q+k+v) heads
        ("Gate/Up", 1024, 5632),    # hidden -> intermediate*2
        ("Down", 2816, 1024),       # intermediate -> hidden
        ("O_proj", 1024, 1024),     # hidden -> hidden
    ]
    for name, K, N in configs:
        weight = torch.randn(N, K, device=device, dtype=torch.bfloat16)
        bias = None
        for M in [1, 4, 8, 16]:
            x = torch.randn(M, K, device=device, dtype=torch.bfloat16)

            # warmup
            for _ in range(3):
                F.linear(x, weight, bias)
                triton_gemv(x, weight, bias)

            r1 = bench(lambda: F.linear(x, weight, bias), f"Linear-{name}", f"M={M}", "cuBLAS")
            r2 = bench(lambda: triton_gemv(x, weight, bias), f"Linear-{name}", f"M={M}", "Triton GEMV")
            results.extend([r1, r2])

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":
    torch.set_default_device("cuda")

    print("=" * 70)
    print("nano-vllm Triton Kernel Optimization Benchmarks")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    benchmark_layernorm()
    benchmark_activation()
    benchmark_rotary()
    benchmark_kvcache()
    benchmark_sampler()
    benchmark_linear()

    dist.destroy_process_group()
    print("\nAll benchmarks complete!")
