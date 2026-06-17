"""
Benchmark script for individual Triton kernel optimizations.
Run on GPU: python bench_kernels.py

Compares the original torch.compile implementations against
the new Triton kernel implementations for each optimized layer.

Use this as microbenchmark evidence, not as a substitute for end-to-end
throughput tests. Most kernels are intended to be numerically equivalent
to the original implementation; the top-k sampler benchmark is an
intentional optional approximation and should not be presented as the
default sampler's semantic-equivalent replacement.
"""

import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.utils.benchmark as benchmark
import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
import torch.distributed as dist

from nanovllm.utils.model_shapes import ModelShapes

# Initialize a single-GPU process group for the linear layers
dist.init_process_group("gloo", rank=0, world_size=1)

SUMMARY = []


def bench(fn, label, sub_label, description, min_run_time=2.0):
    """Helper to create a benchmark Timer."""
    return benchmark.Timer(
        stmt="fn()",
        globals={"fn": fn},
        label=label,
        sub_label=sub_label,
        description=description,
    ).blocked_autorange(min_run_time=min_run_time)


def add_summary(kernel, shape, baseline_name, triton_name, baseline_result, triton_result, correctness="pass", notes=""):
    baseline_us = baseline_result.median * 1e6
    triton_us = triton_result.median * 1e6
    speedup = baseline_us / triton_us if triton_us else float("inf")
    SUMMARY.append({
        "kernel": kernel,
        "shape": shape,
        "baseline": baseline_name,
        "triton": triton_name,
        "baseline_us": baseline_us,
        "triton_us": triton_us,
        "speedup": speedup,
        "correctness": correctness,
        "notes": notes,
    })


def assert_close(name, actual, expected, rtol=2e-2, atol=2e-2):
    actual_f = actual.float()
    expected_f = expected.float()
    torch.testing.assert_close(actual_f, expected_f, rtol=rtol, atol=atol)
    diff = (actual_f - expected_f).abs()
    return f"pass max={diff.max().item():.3g}, mean={diff.mean().item():.3g}"


def benchmark_layernorm(min_run_time, shapes: ModelShapes):
    """Benchmark RMSNorm: original torch.compile vs Triton kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 1: RMSNorm (layernorm.py)")
    print("=" * 70)

    hidden_size = shapes.hidden_size
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
        correctness = assert_close("RMSNorm", new(x.clone()), orig.rms_forward(x.clone()))
        r1 = bench(lambda: orig.rms_forward(x.clone()), "RMSNorm", f"N={N}", "torch.compile", min_run_time)
        r2 = bench(lambda: new(x.clone()), "RMSNorm", f"N={N}", "Triton", min_run_time)
        results.extend([r1, r2])
        add_summary("RMSNorm", f"N={N},D={hidden_size}", "torch.compile", "Triton", r1, r2, correctness)

        # warmup add_rms
        for _ in range(3):
            orig.add_rms_forward(x.clone(), residual.clone())
            new(x.clone(), residual.clone())

        new_out, new_res = new(x.clone(), residual.clone())
        orig_out, orig_res = orig.add_rms_forward(x.clone(), residual.clone())
        assert_close("AddRMSNorm output", new_out, orig_out)
        correctness = assert_close("AddRMSNorm residual", new_res, orig_res)
        r3 = bench(lambda: orig.add_rms_forward(x.clone(), residual.clone()), "AddRMSNorm", f"N={N}", "torch.compile", min_run_time)
        r4 = bench(lambda: new(x.clone(), residual.clone()), "AddRMSNorm", f"N={N}", "Triton", min_run_time)
        results.extend([r3, r4])
        add_summary("AddRMSNorm", f"N={N},D={hidden_size}", "torch.compile", "Triton", r3, r4, correctness)

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_activation(min_run_time, shapes: ModelShapes):
    """Benchmark SiluAndMul: original torch.compile vs Triton kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 2: SiluAndMul (activation.py)")
    print("=" * 70)

    device = "cuda"
    intermediate_size = shapes.intermediate_size

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

        correctness = assert_close("SiluAndMul", new(x.clone()), orig(x.clone()))
        r1 = bench(lambda: orig(x.clone()), "SiluAndMul", f"N={N}", "torch.compile", min_run_time)
        r2 = bench(lambda: new(x.clone()), "SiluAndMul", f"N={N}", "Triton", min_run_time)
        results.extend([r1, r2])
        add_summary("SiluAndMul", f"N={N},D={intermediate_size}", "torch.compile", "Triton", r1, r2, correctness)

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_rotary(min_run_time, shapes: ModelShapes):
    """Benchmark RotaryEmbedding: original torch.compile vs Triton kernel."""
    print("\n" + "=" * 70)
    print("BENCHMARK 3: RotaryEmbedding (rotary_embedding.py)")
    print("=" * 70)

    device = "cuda"
    head_dim = shapes.head_dim
    num_heads = shapes.num_attention_heads
    num_kv_heads = shapes.num_key_value_heads
    max_pos = shapes.max_position_embeddings

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

        new_q, new_k = new(positions, q.clone(), k.clone())
        orig_q, orig_k = orig(positions, q.clone(), k.clone())
        assert_close("RoPE q", new_q, orig_q)
        correctness = assert_close("RoPE k", new_k, orig_k)
        r1 = bench(lambda: orig(positions, q.clone(), k.clone()), "RoPE", f"N={N}", "torch.compile", min_run_time)
        r2 = bench(lambda: new(positions, q.clone(), k.clone()), "RoPE", f"N={N}", "Triton", min_run_time)
        results.extend([r1, r2])
        add_summary("RoPE", f"N={N},QH={num_heads},KVH={num_kv_heads},D={head_dim}", "torch.compile", "Triton", r1, r2, correctness)

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_kvcache(min_run_time, shapes: ModelShapes):
    """Benchmark store_kvcache: original 1D grid vs new 2D grid."""
    print("\n" + "=" * 70)
    print("BENCHMARK 4: store_kvcache (attention.py)")
    print("=" * 70)

    import triton
    import triton.language as tl
    device = "cuda"
    num_kv_heads = shapes.num_key_value_heads
    head_dim = shapes.head_dim
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

    # --- Experimental 2D kernel ---
    from nanovllm.layers.attention import store_kvcache_2d as store_kvcache_new

    results = []
    for N in [1, 16, 128, 512, 2048]:
        key = torch.randn(N, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        value = torch.randn(N, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        k_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        v_cache = torch.zeros(num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=torch.bfloat16)
        k_cache_ref = torch.zeros_like(k_cache)
        v_cache_ref = torch.zeros_like(v_cache)
        # Use unique slots for deterministic correctness checks. Duplicate slots
        # create write races where old 1D and new 2D kernels may legitimately
        # leave different "last writer" values in the same cache location.
        slot_mapping = torch.randperm(num_blocks * block_size, device=device, dtype=torch.int32)[:N]

        store_kvcache_old(key, value, k_cache_ref, v_cache_ref, slot_mapping)
        store_kvcache_new(key, value, k_cache, v_cache, slot_mapping)
        torch.cuda.synchronize()
        assert_close("KVCache k", k_cache, k_cache_ref, rtol=0, atol=0)
        correctness = assert_close("KVCache v", v_cache, v_cache_ref, rtol=0, atol=0)

        # warmup
        for _ in range(3):
            store_kvcache_old(key, value, k_cache, v_cache, slot_mapping)
            store_kvcache_new(key, value, k_cache, v_cache, slot_mapping)

        r1 = bench(lambda: store_kvcache_old(key, value, k_cache, v_cache, slot_mapping), "KVCache", f"N={N}", "1D-grid", min_run_time)
        r2 = bench(lambda: store_kvcache_new(key, value, k_cache, v_cache, slot_mapping), "KVCache", f"N={N}", "2D-grid", min_run_time)
        results.extend([r1, r2])
        add_summary("KVCacheStore", f"N={N},KVH={num_kv_heads},D={head_dim}", "1D-grid", "2D-grid", r1, r2, correctness)

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_sampler(min_run_time, shapes: ModelShapes):
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
    vocab_sizes = [shapes.vocab_size]
    if shapes.vocab_size != 32000:
        vocab_sizes.insert(0, 32000)
    for V in vocab_sizes:
        for N in [1, 16, 64, 256]:
            logits = torch.randn(N, V, device=device, dtype=torch.bfloat16)
            temps = torch.ones(N, device=device, dtype=torch.float32)

            orig = SamplerOriginal()
            new = Sampler(top_k=50)

            # warmup
            for _ in range(3):
                orig(logits.clone(), temps)
                new(logits.clone(), temps)

            r1 = bench(lambda: orig(logits.clone(), temps), "Sampler", f"V={V},N={N}", "torch.compile full-softmax", min_run_time)
            r2 = bench(lambda: new(logits.clone(), temps), "Sampler", f"V={V},N={N}", "Triton top-k", min_run_time)
            results.extend([r1, r2])
            add_summary(
                "SamplerTopK",
                f"N={N},V={V},K=50",
                "torch.compile full-softmax",
                "Triton top-k",
                r1,
                r2,
                correctness="n/a",
                notes="optional approximation; not semantic-equivalent",
            )

    compare = benchmark.Compare(results)
    compare.print()


def benchmark_linear(min_run_time, shapes: ModelShapes):
    """Benchmark Linear: F.linear vs Triton GEMV for small batches."""
    print("\n" + "=" * 70)
    print("BENCHMARK 6: Linear GEMV (linear.py)")
    print("=" * 70)

    device = "cuda"
    from nanovllm.layers.linear import triton_gemv

    results = []
    configs = [
        ("QKV", shapes.hidden_size, shapes.qkv_dim),
        ("O_proj", shapes.o_proj_in, shapes.hidden_size),
        ("Gate/Up", shapes.hidden_size, shapes.intermediate_size * 2),
        ("Down", shapes.intermediate_size, shapes.hidden_size),
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

            correctness = assert_close(
                "Linear GEMV",
                triton_gemv(x, weight, bias),
                F.linear(x, weight, bias),
                rtol=5e-2,
                atol=5e-1,
            )
            r1 = bench(lambda: F.linear(x, weight, bias), f"Linear-{name}", f"M={M}", "cuBLAS", min_run_time)
            r2 = bench(lambda: triton_gemv(x, weight, bias), f"Linear-{name}", f"M={M}", "Triton GEMV", min_run_time)
            results.extend([r1, r2])
            add_summary(f"Linear-{name}", f"M={M},K={K},N={N}", "cuBLAS F.linear", "Triton GEMV", r1, r2, correctness)

    compare = benchmark.Compare(results)
    compare.print()


def print_summary():
    print("\n" + "=" * 70)
    print("Markdown summary table")
    print("=" * 70)
    lines = [
        "| Kernel | Shape | Baseline | Triton | Speedup | Correctness | Notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in SUMMARY:
        lines.append(
            f"| {row['kernel']} | {row['shape']} | "
            f"{row['baseline_us']:.2f} us | {row['triton_us']:.2f} us | "
            f"{row['speedup']:.2f}x | {row['correctness']} | {row['notes']} |"
        )
    text = "\n".join(lines)
    print(text)
    return text


def load_shapes(model_path: str | None) -> ModelShapes:
    if model_path:
        return ModelShapes.from_model(model_path)
    return ModelShapes.default()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Triton kernels against original torch.compile/eager baselines.")
    parser.add_argument("--model", type=str, default=None, help="Optional model path used to derive Qwen layer shapes.")
    parser.add_argument("--min-run-time", type=float, default=2.0)
    parser.add_argument("--skip-sampler", action="store_true", help="Skip the expensive full-vocab sampler benchmark.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save the final Markdown summary table.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for bench_kernels.py")

    torch.set_default_device("cuda")

    print("=" * 70)
    print("nano-vllm Triton Kernel Optimization Benchmarks")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    shapes = load_shapes(args.model)
    print(f"Shapes: {shapes.model_name} ({shapes.describe()})")

    benchmark_layernorm(args.min_run_time, shapes)
    benchmark_activation(args.min_run_time, shapes)
    benchmark_rotary(args.min_run_time, shapes)
    benchmark_kvcache(args.min_run_time, shapes)
    if not args.skip_sampler:
        benchmark_sampler(args.min_run_time, shapes)
    benchmark_linear(args.min_run_time, shapes)
    summary = print_summary()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(summary + "\n", encoding="utf-8")
        print(f"\nSaved Markdown summary to {output_path}")

    dist.destroy_process_group()
    print("\nAll benchmarks complete!")
