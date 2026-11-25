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
from docta_train.fit_args import DEFAULT_LR, build_parser, config_of
from docta_train.guard import protect_machine
from docta_train.model import Config
from docta_train.shortcut import balance, shape, shape_accuracy
from docta_train.training import Batch, run_epoch, validate
from docta_train.words import Vocab

Data = list[tuple[list[int], float]]


def encoded(examples: Sequence[Example], vocab: Vocab, max_len: int) -> Data:
    return [(vocab.encode(e.question, e.excerpts)[:max_len], float(e.label)) for e in examples]


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
    vocab: Vocab | None = None,
    warmup: int = 0,
) -> dict:
    """Trains from scratch, or `epochs` more on OUT/last.pt with `resume`;
    keeps the epoch with the lowest validation loss in OUT/model.pt. `lr`
    None means the default when fresh, the run's own when resumed. `vocab`
    defaults to the one learned from `train`."""
    out.mkdir(parents=True, exist_ok=True)
    # With the vocabulary, a dataset regenerated at the same size is refused too.
    data = {"seed": seed, "batch": batch, "train": len(train), "val": len(val)}
    vocab = vocab or Vocab.build(train)  # from train only: val stays unseen
    run = checkpoint.resume(out / "last.pt", config, vocab, lr, data) if resume else None
    run = run or checkpoint.fresh(config, vocab, DEFAULT_LR if lr is None else lr, seed, warmup)
    train_data = encoded(train, vocab, config.max_len)
    val_data = encoded(val, vocab, config.max_len)
    shapes = [shape(e.question) for e in val]
    start = time.monotonic()
    with (out / "log.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
        for _ in range(epochs):
            metrics = next_epoch(run, train_data, val_data, shapes, batch)
            # Best, then last, then the log: a run killed in between resumes from
            # an epoch whose best is already saved and whose line is not yet logged.
            if run.best is metrics:
                checkpoint.save_best(run, metrics, out / "model.pt")
            checkpoint.save_last(run, out / "last.pt", data)
            log.write(json.dumps({**metrics, "minutes": (time.monotonic() - start) / 60}) + "\n")
            log.flush()
            if max_minutes is not None and time.monotonic() - start > max_minutes * 60:
                break
    return {"best": run.best, "parameters": sum(p.numel() for p in run.model.parameters())}


def next_epoch(
    run: checkpoint.Run, train_data: Data, val_data: Data, shapes: Sequence[str], batch: int
) -> dict:
    """One epoch of training then validation; `run` moves to it."""
    run.epoch += 1
    metrics = {
        "epoch": run.epoch,
        "train_loss": run_epoch(
            run.model, run.optimiser, batches(train_data, batch, run.rng), run.warmup
        ),
    }
    metrics |= validate(run.model, batches(val_data, batch), shapes)
    if run.best is None or metrics["val_loss"] < run.best["val_loss"]:
        run.best = metrics
    return metrics


def load(args: argparse.Namespace) -> tuple[list[Example], list[Example], dict[str, float], Vocab]:
    """Train and val as trained on, and what guessing from the shape scores
    before and after balancing: above that line, the core reads. Caps apply
    first, so balancing is never undone by a cut and both scores see the same
    examples. The vocabulary is learned before balancing, which picks by label."""
    train = list(read_examples(args.data / "train.jsonl"))[: args.max_train]
    val = list(read_examples(args.data / "val.jsonl"))[: args.max_val]
    shortcut = {"unbalanced": shape_accuracy(train, val)}
    vocab = Vocab.build(train)
    if not args.no_balance:
        train, val = balance(train, args.seed), balance(val, args.seed)
    return train, val, shortcut | {"trained_on": shape_accuracy(train, val)}, vocab


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
        train, val, shortcut, vocab = load(args)
    except ValueError as e:
        print(f"error: {e} (balancing drops shapes with one label; see --no-balance)",
              file=sys.stderr)  # fmt: skip
        return 1
    announce(shortcut, train, val)
    try:
        report = fit(
            train, val, args.out, config_of(args), args.epochs, args.batch,
            args.lr, args.seed, args.max_minutes,
            resume=args.resume, vocab=vocab, warmup=args.warmup,
        )  # fmt: skip
    except checkpoint.ResumeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report | {"shape_accuracy": shortcut}, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
