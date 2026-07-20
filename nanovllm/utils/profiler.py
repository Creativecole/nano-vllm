from contextlib import contextmanager, nullcontext
import os

import torch


_TORCH_RANGES_ENABLED = os.environ.get("NANOVLLM_PROFILE_RANGES") == "1"
_NVTX_RANGES_ENABLED = os.environ.get("NANOVLLM_NVTX") == "1"


def configure_profile_ranges(*, torch_ranges=False, nvtx=False):
    """Enable ranges explicitly for profiler entry points in the current process."""
    global _TORCH_RANGES_ENABLED, _NVTX_RANGES_ENABLED
    _TORCH_RANGES_ENABLED = bool(torch_ranges)
    _NVTX_RANGES_ENABLED = bool(nvtx)


def profile_range(name: str):
    """Emit opt-in PyTorch/NVTX ranges without affecting the default runtime."""
    use_torch_profiler = _TORCH_RANGES_ENABLED
    use_nvtx = _NVTX_RANGES_ENABLED and torch.cuda.is_available()
    if not use_torch_profiler and not use_nvtx:
        return nullcontext()
    return _active_profile_range(name, use_torch_profiler, use_nvtx)


@contextmanager
def _active_profile_range(name: str, use_torch_profiler: bool, use_nvtx: bool):
    record = torch.profiler.record_function(name) if use_torch_profiler else None
    if record is not None:
        record.__enter__()
    if use_nvtx:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if use_nvtx:
            torch.cuda.nvtx.range_pop()
        if record is not None:
            record.__exit__(None, None, None)
