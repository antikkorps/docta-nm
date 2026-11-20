"""The command line of `docta_train.fit`: its options and what they default to."""

import argparse
from pathlib import Path

from docta_train import checkpoint
from docta_train.model import Config

DEFAULT_LR = 3e-4


def within(kind: type, low: float, high: float):
    """An argparse type that refuses values outside [low, high]: a negative
    warmup would turn the learning rate negative without any error."""

    def parse(text: str):
        value = kind(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be between {low} and {high}")
        return value

    return parse


def config_of(args: argparse.Namespace) -> Config:
    """The run's own dropout on a resume that names none, like its learning rate."""
    if args.dropout is not None:
        return Config(dropout=args.dropout)
    if not args.resume or not (args.out / "last.pt").exists():
        return Config()  # without last.pt, fit says there is nothing to resume
    saved = checkpoint.load_last(args.out / "last.pt").get("config") or {}
    return Config(dropout=saved.get("dropout", Config.dropout))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m docta_train.fit")
    p.add_argument("--data", type=Path, default=Path("data/evaluate"))
    p.add_argument("--out", type=Path, default=Path("models/core_evaluate"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, help=f"default {DEFAULT_LR}; with --resume, the run's own")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dropout", type=within(float, 0, 0.99), help="default 0; with --resume, the run's own"
    )
    p.add_argument(
        "--warmup",
        type=within(int, 0, float("inf")),
        default=0,
        help="steps of learning-rate climb; a resumed run keeps its own",
    )
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
