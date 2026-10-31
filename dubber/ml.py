"""GPU helpers. The target machine has 4 GB VRAM, so every stage loads its model,
uses it, and releases it before the next stage starts."""

from __future__ import annotations

import gc
from contextlib import contextmanager

from dubber.config import get_settings


def device() -> str:
    import torch

    wanted = get_settings().device
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return wanted


@contextmanager
def gpu_scope():
    """Free cached CUDA memory when the block exits (models must be deleted inside)."""
    try:
        yield
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
