"""The command line of `docta_train.fit`: its options and what they default
to, some shared with `docta_train.fit_read`."""

import argparse
from dataclasses import replace
from pathlib import Path

from docta_train import checkpoint
from docta_train.model import Config

# Over three seeds on val, these got every Evaluate run off the shape-alone
# plateau; warmup 1000 left two seeds in four there, a lower lr overfitted.
DEFAULT_LR = 1e-3
DEFAULT_DROPOUT = 0.1
DEFAULT_WARMUP = 4000  # Read's too: it narrows its spread over seeds (README)


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
    """On a resume, the run's own core: its length is in its weights. A
    named dropout other than the run's own is refused by the resume."""
    if not args.resume:
        return Config(dropout=DEFAULT_DROPOUT if args.dropout is None else args.dropout)
    last = args.out / "last.pt"
    # Without last.pt, fit says there is nothing to resume.
    saved = (checkpoint.load_last(last).get("config") or {}) if last.exists() else {}
    try:
        config = Config(**saved)
    except TypeError as e:  # a setting this code no longer has
        raise checkpoint.ResumeError(f"{last}: {e}: {checkpoint.RERUN}") from e
    return config if args.dropout is None else replace(config, dropout=args.dropout)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m docta_train.fit")
    p.add_argument("--data", type=Path, default=Path("data/evaluate"))
    p.add_argument("--out", type=Path, default=Path("models/core_evaluate"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, help=f"default {DEFAULT_LR}; with --resume, the run's own")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dropout",
        type=within(float, 0, 0.99),
        help=f"default {DEFAULT_DROPOUT}; with --resume, the run's own",
    )
    p.add_argument(
        "--warmup",
        type=within(int, 0, float("inf")),
        default=DEFAULT_WARMUP,
        help=f"learning-rate climb in steps (default {DEFAULT_WARMUP}); a resume keeps its own",
    )
    p.add_argument(
        "--max-train", type=int, help="cap the training examples read (before balancing)"
    )
    p.add_argument(
        "--max-val", type=int, help="cap the validation examples read (before balancing)"
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        help="start no epoch that would likely end past this budget (the first always runs)",
    )
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
