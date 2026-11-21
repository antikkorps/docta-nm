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
from docta_orchestrator.search import search

from docta_train.core_model import CoreChainModel, CoreEvaluateModel
from docta_train.predictor import Predictor
from docta_train.read_model import CoreReadModel
from docta_train.reader import ReadPredictor


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m docta_train.run_core")
    p.add_argument("split", type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path("models/core_evaluate/model.pt"))
    p.add_argument("--out", type=Path, default=Path("predictions"))
    p.add_argument("--traces", type=Path, default=Path("data/traces"))
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument(
        "--reader",
        type=Path,
        default=Path("models/core_read/model.pt"),
        help="a trained Read core; when it exists, core_read runs too",
    )
    args = p.parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL is not set (see .env.example)", file=sys.stderr)
        return 2
    try:
        core = Predictor(args.checkpoint)
        models = [CoreEvaluateModel(core, args.threshold), CoreChainModel(core, args.threshold)]
        if args.reader.exists():
            models.append(CoreReadModel(core, ReadPredictor(args.reader), args.threshold))
        else:
            # `just core` scores every core_*.jsonl: an old run must not pass for this one.
            (args.out / "core_read.jsonl").unlink(missing_ok=True)
        questions = [r for _, r in read_records(args.split) if r["type"] == "question"]
        with psycopg.connect(url, autocommit=True) as conn:
            run_models(
                questions,
                models,
                lambda wid: lambda q: search(conn, wid, q),
                args.out,
                args.traces,
            )
    except (OSError, ValueError, psycopg.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
