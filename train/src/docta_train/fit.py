"""Trains the Evaluate core: `python -m docta_train.fit --data data/evaluate`.

Writes OUT/model.pt (best epoch by validation loss) and OUT/log.jsonl.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from docta_train.evaluate_data import Example, read_examples
from docta_train.guard import protect_machine
from docta_train.model import Config, CoreEvaluate, encode_full
from docta_train.training import Batch, run_epoch, validate

Encoded = list[tuple[list[int], float]]


def encoded(examples: Sequence[Example], config: Config) -> Encoded:
    """Token ids and label per example; refuses to cut any, since a cut example
    can lose the very excerpt its label depends on."""
    full = [encode_full(e.question, e.excerpts) for e in examples]
    cut = [len(ids) for ids in full if len(ids) > config.max_len]
    if cut:
        raise ValueError(
            f"{len(cut)} of {len(full)} examples exceed max_len={config.max_len} tokens"
            f" (longest: {max(cut)}): raise max_len or shorten the excerpts"
        )
    return [(ids, float(e.label)) for ids, e in zip(full, examples, strict=True)]


def batches(data: Encoded, size: int, rng: random.Random | None = None) -> list[Batch]:
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
    lr: float,
    seed: int,
    max_minutes: float | None = None,
) -> dict:
    """Trains from scratch; keeps the epoch with the lowest validation loss."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    model = CoreEvaluate(config)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    train_data, val_data = encoded(train, config), encoded(val, config)
    best: dict | None = None
    start = time.monotonic()
    with (out / "log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            metrics = {
                "epoch": epoch,
                "train_loss": run_epoch(model, optimiser, batches(train_data, batch, rng)),
            }
            metrics |= validate(model, batches(val_data, batch))
            log.write(json.dumps({**metrics, "minutes": (time.monotonic() - start) / 60}) + "\n")
            log.flush()
            if best is None or metrics["val_loss"] < best["val_loss"]:
                best = metrics
                save(model, best, out / "model.pt")
            if max_minutes is not None and time.monotonic() - start > max_minutes * 60:
                break
    return {"best": best, "parameters": sum(p.numel() for p in model.parameters())}


def save(model: CoreEvaluate, metrics: dict, path: Path) -> None:
    torch.save(
        {"config": asdict(model.config), "state": model.state_dict(), "metrics": metrics}, path
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m docta_train.fit")
    p.add_argument("--data", type=Path, default=Path("data/evaluate"))
    p.add_argument("--out", type=Path, default=Path("models/core_evaluate"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, help="cap the number of training examples")
    p.add_argument("--max-val", type=int, help="cap validation examples: shorter epochs")
    p.add_argument("--max-minutes", type=float, help="stop after the epoch that passes this budget")
    p.add_argument("--max-gb", type=float, default=6.0, help="address-space cap (Linux only)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads left to training")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    protect_machine(max_gb=args.max_gb, threads=args.threads)
    train = list(read_examples(args.data / "train.jsonl"))[: args.max_train]
    val = list(read_examples(args.data / "val.jsonl"))[: args.max_val]
    report = fit(
        train,
        val,
        args.out,
        Config(),
        args.epochs,
        args.batch,
        args.lr,
        args.seed,
        args.max_minutes,
    )
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
