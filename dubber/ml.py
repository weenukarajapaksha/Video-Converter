"""GPU helpers. The target machine has 4 GB VRAM, so every stage loads its model,
uses it, and releases it before the next stage starts."""

from __future__ import annotations

import gc
from contextlib import contextmanager

from dubber.config import get_settings


def device() -> str:
    # Importing torch also loads its bundled CUDA runtime (cuBLAS/cuDNN DLLs), which
    # CTranslate2 (faster-whisper) then reuses - no separate CUDA toolkit needed.
    import torch

    wanted = get_settings().device
    if wanted == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return wanted


@contextmanager
def gpu_scope():
    """Exclusive GPU access across processes, and free cached CUDA memory on exit
    (models must be deleted inside the block).

    With 4 GB VRAM, two processes running models at once (e.g. the web server and a
    CLI run) spill into shared system memory and slow down ~100x, so GPU work is
    serialized with a lock file.
    """
    from filelock import FileLock

    lock_dir = get_settings().jobs_dir
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_dir / ".gpu.lock"))
    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
