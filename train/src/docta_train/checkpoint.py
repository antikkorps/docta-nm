"""What a training run needs to stop and go on: weights, optimiser, shuffling.

OUT/model.pt holds the best epoch, for serving. OUT/last.pt holds the last one
with everything a resumed run needs to continue exactly as if it never
stopped, so a promising run can get more epochs instead of starting over.
"""

import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from docta_train.model import Config, CoreEvaluate

RERUN = "rerun without --resume"


@dataclass
class Run:
    model: CoreEvaluate
    optimiser: torch.optim.Optimizer
    rng: random.Random  # shuffles training batches
    epoch: int = 0  # last finished epoch
    best: dict | None = None


def fresh(config: Config, lr: float, seed: int) -> Run:
    torch.manual_seed(seed)
    model = CoreEvaluate(config)
    return Run(model, torch.optim.AdamW(model.parameters(), lr=lr), random.Random(seed))


def save_best(model: CoreEvaluate, metrics: dict, path: Path) -> None:
    torch.save(
        {"config": asdict(model.config), "state": model.state_dict(), "metrics": metrics}, path
    )


def save_last(run: Run, path: Path, data: dict) -> None:
    """`data` names what was trained on, so a resume on other data is refused."""
    torch.save(
        {
            "config": asdict(run.model.config),
            "state": run.model.state_dict(),
            "optimiser": run.optimiser.state_dict(),
            "rng": run.rng.getstate(),
            "epoch": run.epoch,
            "best": run.best,
            "data": data,
        },
        path,
    )


def resume(path: Path, config: Config, lr: float, data: dict) -> Run:
    """The run saved at `path`, with `lr` as its new learning rate."""
    if not path.exists():
        raise ValueError(f"nothing to resume: no {path}")
    saved = torch.load(path, weights_only=True)
    if saved["config"] != asdict(config) or saved["data"] != data:
        raise ValueError(f"{path} was trained with another core or other data: {RERUN}")
    run = fresh(config, lr, seed=0)
    run.model.load_state_dict(saved["state"])
    run.optimiser.load_state_dict(saved["optimiser"])
    for group in run.optimiser.param_groups:
        group["lr"] = lr
    run.rng.setstate(saved["rng"])
    run.epoch, run.best = saved["epoch"], saved["best"]
    return run
