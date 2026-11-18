"""Training must never take the whole machine down: each check runs in a child
process, because the limits it sets cannot be lifted once applied."""

import subprocess
import sys

import pytest
from docta_train import guard

PROTECTED_STEP = """
import torch
from docta_train.guard import protect_machine
from docta_train.model import Config, CoreEvaluate
from docta_train.training import train_steps
protect_machine(max_gb={max_gb}, threads=1)
seqs = [[5] * 1000 for _ in range({batch})]
train_steps(CoreEvaluate(Config()), [(seqs, torch.zeros({batch}))], 1e-3)
"""


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@pytest.mark.skipif(sys.platform != "linux", reason="RLIMIT_AS is only enforced on Linux")
def test_guard_caps_memory_threads_and_priority():
    child = run(
        "import os, torch\n"
        "from docta_train.guard import protect_machine\n"
        "protect_machine(max_gb=1, threads=1)\n"
        "assert torch.get_num_threads() == 1 and os.nice(0) >= 10\n"
        "try:\n"
        "    bytearray(2 << 30)\n"
        "except MemoryError:\n"
        "    print('refused')\n"
    )
    assert child.returncode == 0, child.stderr
    assert child.stdout.strip() == "refused"


def test_default_core_trains_long_sequences_in_little_memory():
    # Attention dropout forces PyTorch's CPU attention to keep every
    # length x length matrix for backward (~250 MB per 1000-byte example):
    # batch 64 needed ~16 GB and froze the machine.
    child = run(PROTECTED_STEP.format(max_gb=2, batch=8))
    assert child.returncode == 0, child.stderr[-500:]


def test_guard_skips_the_memory_cap_where_the_kernel_ignores_it(monkeypatch, capsys):
    # macOS accepts or rejects RLIMIT_AS without enforcing it: no false safety.
    monkeypatch.setattr(guard.sys, "platform", "darwin")
    monkeypatch.setattr(guard.resource, "setrlimit", lambda *a: pytest.fail("cap set"))
    monkeypatch.setattr(guard.os, "nice", lambda n: 0)
    threads = guard.torch.get_num_threads()
    try:
        guard.protect_machine(max_gb=1, threads=1)
    finally:
        guard.torch.set_num_threads(threads)
    assert "no memory cap" in capsys.readouterr().err
