"""Trains the Evaluate core: `python -m docta_train.fit --data data/evaluate`.

Writes OUT/model.pt (best epoch by validation loss), OUT/last.pt (to resume)
and OUT/log.jsonl. `--resume` adds `--epochs` more to the run in OUT.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from docta_train import checkpoint
from docta_train.evaluate_data import Example, read_examples
from docta_train.guard import protect_machine
from docta_train.model import Config, Encoded, encode
from docta_train.shortcut import balance, shape_accuracy
from docta_train.training import Batch, run_epoch, validate

Data = list[tuple[Encoded, float]]
DEFAULT_LR = 3e-4


def encoded(examples: Sequence[Example], config: Config) -> Data:
    return [(encode(e.question, e.excerpts, config.max_len), float(e.label)) for e in examples]


def batches(data: Data, size: int, rng: random.Random | None = None) -> list[Batch]:
    """Fixed-size batches, shuffled by `rng` when given (training)."""
    order = list(range(len(data)))
    if rng is not None:
        rng.shuffle(order)
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    return [([data[i][0] for i in c], torch.tensor([data[i][1] for i in c])) for c in chunks]


def fit(
    train: Sequence[Example],
    val: Sequence[Example],
    out: Path,
    config: Config,
    epochs: int,
    batch: int,
    lr: float | None,
    seed: int,
    max_minutes: float | None = None,
    resume: bool = False,
) -> dict:
    """Trains from scratch, or `epochs` more on OUT/last.pt with `resume`;
    keeps the epoch with the lowest validation loss in OUT/model.pt. `lr`
    None means the default when fresh, the run's own when resumed."""
    out.mkdir(parents=True, exist_ok=True)
    # Sizes, not contents: a dataset regenerated at the same size would resume.
    data = {"seed": seed, "batch": batch, "train": len(train), "val": len(val)}
    run = checkpoint.resume(out / "last.pt", config, lr, data) if resume else None
    run = run or checkpoint.fresh(config, DEFAULT_LR if lr is None else lr, seed)
    train_data, val_data = encoded(train, config), encoded(val, config)
    start = time.monotonic()
    with (out / "log.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
        for _ in range(epochs):
            metrics = next_epoch(run, train_data, val_data, batch)
            # Best, then last, then the log: a run killed in between resumes from
            # an epoch whose best is already saved and whose line is not yet logged.
            if run.best is metrics:
                checkpoint.save_best(run.model, metrics, out / "model.pt")
            checkpoint.save_last(run, out / "last.pt", data)
            log.write(json.dumps({**metrics, "minutes": (time.monotonic() - start) / 60}) + "\n")
            log.flush()
            if max_minutes is not None and time.monotonic() - start > max_minutes * 60:
                break
    return {"best": run.best, "parameters": sum(p.numel() for p in run.model.parameters())}


def next_epoch(run: checkpoint.Run, train_data: Data, val_data: Data, batch: int) -> dict:
    """One epoch of training then validation; `run` moves to it."""
    run.epoch += 1
    metrics = {
        "epoch": run.epoch,
        "train_loss": run_epoch(run.model, run.optimiser, batches(train_data, batch, run.rng)),
    }
    metrics |= validate(run.model, batches(val_data, batch))
    if run.best is None or metrics["val_loss"] < run.best["val_loss"]:
        run.best = metrics
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m docta_train.fit")
    p.add_argument("--data", type=Path, default=Path("data/evaluate"))
    p.add_argument("--out", type=Path, default=Path("models/core_evaluate"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, help=f"default {DEFAULT_LR}; with --resume, the run's own")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-train", type=int, help="cap the training examples read (before balancing)"
    )
    p.add_argument(
        "--max-val", type=int, help="cap the validation examples read (before balancing)"
    )
    p.add_argument("--max-minutes", type=float, help="stop after the epoch that passes this budget")
    p.add_argument("--max-gb", type=float, default=6.0, help="address-space cap (Linux only)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads left to training")
    p.add_argument(
        "--no-balance",
        action="store_true",
        help="keep every example, even where the question's shape gives the label away",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="train --epochs more on OUT/last.pt instead of starting from scratch",
    )
    return p


def load(args: argparse.Namespace) -> tuple[list[Example], list[Example], dict[str, float]]:
    """Train and val as trained on, and what guessing from the shape scores
    before and after balancing: above that line, the core reads. Caps apply
    first, so balancing is never undone by a cut and both scores see the same
    examples."""
    train = list(read_examples(args.data / "train.jsonl"))[: args.max_train]
    val = list(read_examples(args.data / "val.jsonl"))[: args.max_val]
    shortcut = {"unbalanced": shape_accuracy(train, val)}
    if not args.no_balance:
        train, val = balance(train, args.seed), balance(val, args.seed)
    return train, val, shortcut | {"trained_on": shape_accuracy(train, val)}


def announce(shortcut: dict[str, float], train: Sequence[Example], val: Sequence[Example]) -> None:
    # Printed before hours of training, so a run can be judged from its first log line.
    print(
        f"shape alone on val: {shortcut['unbalanced']:.3f} unbalanced, "
        f"{shortcut['trained_on']:.3f} as trained on ({len(train)} train, {len(val)} val)",
        file=sys.stderr,
    )


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    protect_machine(max_gb=args.max_gb, threads=args.threads)
    try:
        train, val, shortcut = load(args)
    except ValueError as e:
        print(f"error: {e} (balancing drops shapes with one label; see --no-balance)",
              file=sys.stderr)  # fmt: skip
        return 1
    announce(shortcut, train, val)
    try:
        report = fit(
            train, val, args.out, Config(), args.epochs, args.batch, args.lr, args.seed,
            args.max_minutes, resume=args.resume,
        )  # fmt: skip
    except checkpoint.ResumeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report | {"shape_accuracy": shortcut}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
