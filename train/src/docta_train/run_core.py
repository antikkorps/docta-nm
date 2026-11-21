"""Runs the trained core on a split, next to the fake models' predictions.

`python -m docta_train.run_core data/test.jsonl --checkpoint models/core_evaluate/model.pt`
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
from docta_orchestrator.baseline import run_models
from docta_orchestrator.jsonl import read_records
from docta_orchestrator.protocol import run_on_split
from docta_orchestrator.search import search

from docta_train.core_model import CoreChainModel, CoreEvaluateModel
from docta_train.predictor import Predictor
from docta_train.read_model import CoreReadModel
from docta_train.reader import ReadPredictor


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m docta_train.run_core")
    p.add_argument("split", type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path("models/core_evaluate/model.pt"))
    p.add_argument("--out", type=Path, default=Path("predictions"))
    p.add_argument("--traces", type=Path, default=Path("data/traces"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--reader", type=Path, default=Path("models/core_read/model.pt"),
                   help="a trained Read core; when it exists, core_read runs too")  # fmt: skip
    p.add_argument("--human", type=Path,
                   help="run the human-correction protocol into this directory")  # fmt: skip
    return p.parse_args(argv)


def build_models(args: argparse.Namespace) -> list:
    core = Predictor(args.checkpoint)
    models = [CoreEvaluateModel(core, args.threshold), CoreChainModel(core, args.threshold)]
    if args.reader.exists():
        models.append(CoreReadModel(core, ReadPredictor(args.reader), args.threshold))
    else:
        # `just core` scores every core_*.jsonl: an old run must not pass for this one.
        (args.out / "core_read.jsonl").unlink(missing_ok=True)
    return models


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL is not set (see .env.example)", file=sys.stderr)
        return 2
    try:
        models = build_models(args)
        if args.human:
            # The chain models only: the protocol corrects a bridge mid-way.
            run_on_split(args.split, models[1:], args.human)
            return 0
        questions = [r for _, r in read_records(args.split) if r["type"] == "question"]
        with psycopg.connect(url, autocommit=True) as conn:
            search_for = lambda wid: lambda q: search(conn, wid, q)  # noqa: E731
            run_models(questions, models, search_for, args.out, args.traces)
    except (OSError, ValueError, psycopg.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
