"""Keeps a training run from freezing the machine it shares with other work.

An out-of-memory run under WSL drags Windows down with it, so the process caps
itself: past the cap PyTorch raises an allocation error (RuntimeError or
MemoryError) instead of swapping the host to death.
"""

import os
import resource
import sys

import torch


def protect_machine(max_gb: float, threads: int) -> None:
    """Caps address space and CPU threads, and yields CPU to other programs."""
    cap_memory(max_gb)
    torch.set_num_threads(threads)
    os.nice(10)


def cap_memory(max_gb: float) -> None:
    """The cap counts address space, not RAM: `import torch` alone reserves
    about 1 GB of it, so the usable budget is roughly max_gb - 1."""
    if sys.platform != "linux":
        # macOS accepts or rejects RLIMIT_AS but never enforces it.
        print(f"guard: no memory cap on {sys.platform}, watch memory yourself", file=sys.stderr)
        return
    cap = int(max_gb * (1 << 30))
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
