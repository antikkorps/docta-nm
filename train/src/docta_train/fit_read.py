"""Trains the Read core: `python -m docta_train.fit_read --data data/read`.

Writes OUT/model.pt (the epoch with the best val exactness) and
OUT/log.jsonl, with val exactness per pointer (`val_exact_answer`,
`val_exact_bridge`). Taught retrieve and compare only, val saturated at 100%
from epoch 1; with chains and phrasings of its own it no longer does, and
peaks before the last epoch, so it picks the core kept.
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

from docta_train.fit_args import DEFAULT_WARMUP, within
from docta_train.guard import protect_machine
from docta_train.model import Config, pad_batch, whole
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
from docta_train.training import Warmup, by_shape, over_budget, require
from docta_train.words import Vocab

Item = tuple[list[int], list[int], bool]  # token ids, positions of the answer, bridge head


def encoded(examples: Sequence[ReadExample], vocab: Vocab, max_len: int) -> list[Item]:
    every = whole([vocab.encode(e.question, e.excerpts) for e in examples], max_len)
    items = []
    for e, ids in zip(examples, every, strict=True):
        at = answer_positions(aligned(e.question, e.excerpts), e.answer, e.sources, max_len)
        items.append((ids, at, e.head == "bridge"))
    return items


def scored(
    model: CoreRead, items: Sequence[Item]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scores, answer positions and candidate positions for one batch, in
    pointer_loss's order."""
    ids, pad = pad_batch([i for i, _, _ in items])
    target = torch.zeros_like(ids, dtype=torch.bool)
    for row, (_, at, _) in enumerate(items):
        target[row, at] = True
    bridge = torch.tensor([b for _, _, b in items])
    return model(ids, pad, bridge), target, candidates(ids)


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
    reachable = 0
    for start in range(0, len(items), batch):
        scores, target, allowed = scored(model, items[start : start + batch])
        # An answer cut off or absent has no probability to sum: it counts as
        # wrong below, never as an infinite loss.
        has = target.any(1)
        if has.any():
            loss += pointer_loss(scores[has], target[has], allowed[has]).item() * int(has.sum())
        reachable += int(has.sum())
        best = scores.masked_fill(~allowed, float("-inf")).argmax(1).tolist()
        for e, at in zip(val[start : start + batch], best, strict=True):
            right.append(extract(aligned(e.question, e.excerpts), at)[0] == e.answer)
    return {
        "val_loss": loss / max(reachable, 1),
        "val_exact": sum(right) / len(right),
        **{f"val_exact_{head}": exact(right, val, head) for head in ("answer", "bridge")},
        "val_by_shape": by_shape(right, [shape(e.question) for e in val]),
    }


def exact(right: list[bool], val: Sequence[ReadExample], head: str) -> float | None:
    """Exact answers among one pointer's examples: each head is judged on its own job."""
    mine = [r for r, e in zip(right, val, strict=True) if e.head == head]
    return sum(mine) / len(mine) if mine else None


def fit_read(train, val, out: Path, config: Config, epochs: int, batch: int, lr: float,
             seed: int, warmup: int = 0, max_minutes: float | None = None) -> dict:  # fmt: skip
    """Trains from scratch; keeps the epoch with the best val exactness."""
    out.mkdir(parents=True, exist_ok=True)
    run, vocab, train_items, val_items = prepared(train, val, config, lr, seed, warmup)
    model = run["model"]
    metrics, best, start = {}, {}, time.monotonic()
    with (out / "log.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            began = time.monotonic()
            metrics = {"epoch": epoch, "train_loss": run_epoch(run, train_items, batch)}
            metrics |= validate(model, val, val_items, batch)
            # val holds phrasings of its own and peaks before the last epoch
            # (the core memorises train's): it picks the core kept.
            if metrics["val_exact"] > best.get("val_exact", -1.0):
                best = metrics
                save(model, vocab, metrics, out / "model.pt")
            log.write(json.dumps({**metrics, "minutes": (time.monotonic() - start) / 60}) + "\n")
            log.flush()
            now = time.monotonic()
            if over_budget(now - start, now - began, max_minutes):
                break
    return {"last": metrics, "best": best, "parameters": sum(p.numel() for p in model.parameters())}


def prepared(train, val, config: Config, lr: float, seed: int, warmup: int) -> tuple:
    """A fresh core, its optimiser and the encoded splits; refuses a split
    with nothing in it, or a train split with no answer the core can point at
    (an epoch's mean loss would divide by zero)."""
    require(train, "train")
    require(val, "val")
    torch.manual_seed(seed)
    vocab = Vocab.build(train)  # from train only: val stays unseen
    model = CoreRead(config, len(vocab.words))
    run = {"model": model, "optimiser": torch.optim.AdamW(model.parameters(), lr=lr),
           "rng": random.Random(seed), "warmup": Warmup(lr, warmup)}  # fmt: skip
    train_items = encoded(train, vocab, config.max_len)
    require([item for item in train_items if item[1]], "train")
    return run, vocab, train_items, encoded(val, vocab, config.max_len)


def save(model: CoreRead, vocab: Vocab, metrics: dict, path: Path) -> None:
    # "kind" keeps an Evaluate predictor from loading these weights, whose
    # tensors have the very same shapes.
    state = {"kind": "read", "config": model.config.__dict__, "vocab": list(vocab.words)}
    torch.save(state | {"state": model.state_dict(), "metrics": metrics}, path)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m docta_train.fit_read")
    p.add_argument("--data", type=Path, default=Path("data/read"))
    p.add_argument("--out", type=Path, default=Path("models/core_read"))
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=within(int, 0, float("inf")), default=DEFAULT_WARMUP)
    p.add_argument("--dropout", type=within(float, 0, 0.99), default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-train", type=int, help="cap the training examples read")
    p.add_argument(
        "--max-minutes",
        type=float,
        help="start no epoch that would likely end past this budget (the first always runs)",
    )
    p.add_argument("--max-gb", type=float, default=6.0, help="address-space cap (Linux only)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads left to training")
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    protect_machine(max_gb=args.max_gb, threads=args.threads)
    try:
        train = list(read_reads(args.data / "train.jsonl"))[: args.max_train]
        val = list(read_reads(args.data / "val.jsonl"))
    except (OSError, ValueError) as e:
        print(f"error: {e} (run `just read-data` first)", file=sys.stderr)
        return 1
    try:
        report = fit_read(
            train, val, args.out, Config(dropout=args.dropout), args.epochs,
            args.batch, args.lr, args.seed, args.warmup, args.max_minutes,
        )  # fmt: skip
    except (OSError, ValueError) as e:  # --out unwritable, examples too long, a split empty
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
