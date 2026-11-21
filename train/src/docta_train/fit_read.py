"""Trains the Read core: `python -m docta_train.fit_read --data data/read`.

Writes OUT/model.pt (the last epoch) and OUT/log.jsonl. Val is no guide to
the epoch: it saturates at 100% from epoch 1 while ood still moves (best at
epoch 2 over three seeds), so the run length is the choice.
No resume yet: a run is short enough to restart.
"""

import argparse
import json
import random
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from docta_train.fit_args import within
from docta_train.guard import protect_machine
from docta_train.model import Config, pad_batch
from docta_train.read_data import ReadExample, read_reads
from docta_train.reader import (
    CoreRead,
    aligned,
    answer_positions,
    candidates,
    extract,
    pointer_loss,
)
from docta_train.shortcut import shape
from docta_train.training import Warmup, by_shape
from docta_train.words import Vocab

Item = tuple[list[int], list[int]]  # token ids, positions of the answer


def encoded(examples: Sequence[ReadExample], vocab: Vocab, max_len: int) -> list[Item]:
    items = []
    for e in examples:
        ids = vocab.encode(e.question, e.excerpts)[:max_len]
        items.append(
            (ids, answer_positions(aligned(e.question, e.excerpts), e.answer, e.sources, max_len))
        )
    return items


def scored(
    model: CoreRead, items: Sequence[Item]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scores, answer positions and candidate positions for one batch, in
    pointer_loss's order."""
    ids, pad = pad_batch([i for i, _ in items])
    target = torch.zeros_like(ids, dtype=torch.bool)
    for row, (_, at) in enumerate(items):
        target[row, at] = True
    return model(ids, pad), target, candidates(ids)


def run_epoch(run: dict, items: list[Item], batch: int) -> float:
    run["model"].train()
    order = [i for i in range(len(items)) if items[i][1]]  # no answer read: nothing to learn
    run["rng"].shuffle(order)
    losses = []
    for start in range(0, len(order), batch):
        for group in run["optimiser"].param_groups:
            group["lr"] = run["warmup"].next()
        loss = pointer_loss(*scored(run["model"], [items[i] for i in order[start : start + batch]]))
        run["optimiser"].zero_grad()
        loss.backward()
        run["optimiser"].step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


@torch.no_grad()
def validate(model: CoreRead, val: Sequence[ReadExample], items: list[Item], batch: int) -> dict:
    """Loss, and the share of answers copied exactly: what Conclude would state."""
    model.eval()
    right, loss = [], 0.0
    for start in range(0, len(items), batch):
        scores, target, allowed = scored(model, items[start : start + batch])
        loss += pointer_loss(scores, target, allowed).item() * len(scores)
        best = scores.masked_fill(~allowed, float("-inf")).argmax(1).tolist()
        for e, at in zip(val[start : start + batch], best, strict=True):
            right.append(extract(aligned(e.question, e.excerpts), at)[0] == e.answer)
    return {
        "val_loss": loss / len(items),
        "val_exact": sum(right) / len(right),
        "val_by_shape": by_shape(right, [shape(e.question) for e in val]),
    }


def fit_read(train, val, out: Path, config: Config, epochs: int, batch: int, lr: float,
             seed: int, warmup: int = 0, max_minutes: float | None = None) -> dict:  # fmt: skip
    """Trains from scratch; keeps the last epoch."""
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    vocab = Vocab.build(train)  # from train only: val stays unseen
    model = CoreRead(config, len(vocab.words))
    run = {"model": model, "optimiser": torch.optim.AdamW(model.parameters(), lr=lr),
           "rng": random.Random(seed), "warmup": Warmup(lr, warmup)}  # fmt: skip
    train_items, val_items = (
        encoded(train, vocab, config.max_len),
        encoded(val, vocab, config.max_len),
    )
    metrics, start = {}, time.monotonic()
    with (out / "log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            metrics = {"epoch": epoch, "train_loss": run_epoch(run, train_items, batch)}
            metrics |= validate(model, val, val_items, batch)
            save(model, vocab, metrics, out / "model.pt")
            log.write(json.dumps({**metrics, "minutes": (time.monotonic() - start) / 60}) + "\n")
            log.flush()
            if max_minutes is not None and time.monotonic() - start > max_minutes * 60:
                break
    return {"last": metrics, "parameters": sum(p.numel() for p in model.parameters())}


def save(model: CoreRead, vocab: Vocab, metrics: dict, path: Path) -> None:
    # "kind" keeps an Evaluate predictor from loading these weights, whose
    # tensors have the very same shapes.
    state = {"kind": "read", "config": model.config.__dict__, "vocab": list(vocab.words)}
    torch.save(state | {"state": model.state_dict(), "metrics": metrics}, path)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m docta_train.fit_read")
    p.add_argument("--data", type=Path, default=Path("data/read"))
    p.add_argument("--out", type=Path, default=Path("models/core_read"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=within(int, 0, float("inf")), default=1000)
    p.add_argument("--dropout", type=within(float, 0, 0.99), default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, help="cap the training examples read")
    p.add_argument("--max-minutes", type=float, help="stop after the epoch that passes this budget")
    p.add_argument("--max-gb", type=float, default=6.0, help="address-space cap (Linux only)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads left to training")
    args = p.parse_args(argv)
    protect_machine(max_gb=args.max_gb, threads=args.threads)
    try:
        train = list(read_reads(args.data / "train.jsonl"))[: args.max_train]
        val = list(read_reads(args.data / "val.jsonl"))
    except (OSError, ValueError) as e:
        print(f"error: {e} (run `just read-data` first)", file=sys.stderr)
        return 1
    report = fit_read(train, val, args.out, Config(dropout=args.dropout), args.epochs, args.batch,
                      args.lr, args.seed, args.warmup, args.max_minutes)  # fmt: skip
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
