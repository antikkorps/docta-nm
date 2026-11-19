"""What a training run needs to stop and go on: weights, optimiser, shuffling.

OUT/model.pt holds the best epoch, for serving. OUT/last.pt holds the last one
with everything a resumed run needs to continue exactly as if it never
stopped, so a promising run can get more epochs instead of starting over.
"""

import os
import pickle
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from docta_train.model import Config, CoreEvaluate

RERUN = "rerun without --resume"


class ResumeError(ValueError):
    """A resume that cannot continue the saved run."""


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
    """`data` names what was trained on, so a resume on other data is refused.
    Written aside then renamed: a run killed mid-write keeps its previous last.pt."""
    state = {
        "config": asdict(run.model.config),
        "state": run.model.state_dict(),
        "optimiser": run.optimiser.state_dict(),
        "rng": run.rng.getstate(),
        "torch_rng": torch.get_rng_state(),  # dropout draws from it
        "epoch": run.epoch,
        "best": run.best,
        "data": data,
    }
    partial = path.with_suffix(".partial")
    torch.save(state, partial)
    os.replace(partial, path)


def load_last(path: Path) -> dict:
    if not path.exists():
        raise ResumeError(f"nothing to resume: no {path}")
    try:
        return torch.load(path, weights_only=True)
    except (pickle.UnpicklingError, RuntimeError, EOFError) as e:
        raise ResumeError(f"{path} cannot be read ({e.__class__.__name__}): {RERUN}") from e


def resume(path: Path, config: Config, lr: float | None, data: dict) -> Run:
    """The run saved at `path`; `lr` replaces its learning rate when given."""
    saved = load_last(path)
    if saved.get("config") != asdict(config) or saved.get("data") != data:
        raise ResumeError(f"{path} was trained with another core or other data: {RERUN}")
    model = CoreEvaluate(config)
    model.load_state_dict(saved["state"])
    optimiser = torch.optim.AdamW(model.parameters())
    optimiser.load_state_dict(saved["optimiser"])
    for group in optimiser.param_groups:
        group["lr"] = group["lr"] if lr is None else lr
    rng = random.Random()
    rng.setstate(saved["rng"])
    torch.set_rng_state(saved["torch_rng"])
    return Run(model, optimiser, rng, saved["epoch"], saved["best"])
