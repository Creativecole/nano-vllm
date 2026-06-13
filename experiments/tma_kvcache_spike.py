import argparse
import importlib
import random

import torch
import torch.utils.benchmark as benchmark


def has_blackwell_tma_api() -> bool:
    try:
        importlib.import_module("triton.experimental.gluon.language.nvidia.blackwell.tma")
        return True
    except Exception:
        return False


def make_slots(n_tokens: int, num_slots: int, pattern: str):
    if pattern == "contiguous":
        start = random.randint(0, num_slots - n_tokens - 1)
        return torch.arange(start, start + n_tokens, device="cuda", dtype=torch.int32)
    if pattern == "block_local":
        block_size = 256
        blocks = torch.randint(0, num_slots // block_size, (n_tokens,), device="cuda", dtype=torch.int32)
        offsets = torch.randint(0, block_size, (n_tokens,), device="cuda", dtype=torch.int32)
        return blocks * block_size + offsets
    if pattern == "random":
        return torch.randint(0, num_slots, (n_tokens,), device="cuda", dtype=torch.int32)
    raise ValueError(pattern)


def locality_score(slots: torch.Tensor) -> float:
    slots_cpu = slots.cpu().tolist()
    if len(slots_cpu) < 2:
        return 1.0
    adjacent = sum(abs(a - b) == 1 for a, b in zip(slots_cpu, slots_cpu[1:]))
    return adjacent / (len(slots_cpu) - 1)


def main():
    parser = argparse.ArgumentParser(description="Feasibility spike for paged KV cache store vs Blackwell TMA.")
    parser.add_argument("--n-tokens", type=int, default=2048)
    parser.add_argument("--num-blocks", type=int, default=1024)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--min-run-time", type=float, default=1.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment.")

    from nanovllm.layers.attention import store_kvcache

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Capability: sm{torch.cuda.get_device_capability()[0]}{torch.cuda.get_device_capability()[1]}")
    print(f"Blackwell TMA Python API available: {has_blackwell_tma_api()}")
    print()
    print("| pattern | adjacent-slot locality | store latency |")
    print("|---|---:|---:|")

    key = torch.randn(args.n_tokens, args.num_kv_heads, args.head_dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    k_cache = torch.empty(args.num_blocks, args.block_size, args.num_kv_heads, args.head_dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)
    num_slots = args.num_blocks * args.block_size

    for pattern in ["contiguous", "block_local", "random"]:
        slots = make_slots(args.n_tokens, num_slots, pattern)
        for _ in range(5):
            store_kvcache(key, value, k_cache, v_cache, slots)
        torch.cuda.synchronize()
        result = benchmark.Timer(
            stmt="store_kvcache(key, value, k_cache, v_cache, slots)",
            globals={
                "store_kvcache": store_kvcache,
                "key": key,
                "value": value,
                "k_cache": k_cache,
                "v_cache": v_cache,
                "slots": slots,
            },
        ).blocked_autorange(min_run_time=args.min_run_time)
        print(f"| {pattern} | {locality_score(slots):.3f} | {result.median * 1e6:.2f} us |")

    print(
        "\nInterpretation: paged KV writes are slot_mapping-driven scatters. "
        "Use TMA only if real workloads show enough tile locality to amortize "
        "descriptor/setup cost; otherwise prioritize the attention read path."
    )


if __name__ == "__main__":
    main()
