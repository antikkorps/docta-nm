"""Runs the baseline models over a split and writes their predictions and traces.

`python -m docta_orchestrator.baseline data/test.jsonl`. A model only ever
receives the question text and the search tool: never the gold fields.
"""

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

import psycopg

from docta_orchestrator.jsonl import read_records
from docta_orchestrator.loop import Search, run
from docta_orchestrator.models import (
    AlwaysAbstainModel,
    ChallengeModel,
    DigModel,
    KeywordModel,
    MajorityModel,
    RandomModel,
    RuleEvaluateModel,
)
from docta_orchestrator.search import search
from docta_orchestrator.states import Model


def predict(question: dict, model: Model, search: Search, max_rounds: int = 3) -> tuple[dict, dict]:
    """One question through the loop: its prediction record and its trace."""
    outcome, states = run(question["text"], model, search, max_rounds)
    prediction = {
        "question_id": question["question_id"],
        "world_id": question["world_id"],
        "abstained": outcome.exit is not None,
        "answer": outcome.answer,
        "cited_sources": list(outcome.cited),
        "confidence": outcome.confidence,
        "exit": outcome.exit,
    }
    trace = {
        "world_id": question["world_id"],
        "question_id": question["question_id"],
        "model": model.name,
        "question": question["text"],
        "states": states,
        "decision": prediction,
    }
    return prediction, trace


def run_models(
    questions: list[dict],
    models: list[Model],
    search_for: Callable[[str], Search],
    out: Path,
    traces: Path,
) -> None:
    """Writes out/<model>.jsonl and traces/<model>.jsonl, one line per question."""
    out.mkdir(parents=True, exist_ok=True)
    traces.mkdir(parents=True, exist_ok=True)
    for model in models:
        with (
            (out / f"{model.name}.jsonl").open("w", encoding="utf-8") as preds,
            (traces / f"{model.name}.jsonl").open("w", encoding="utf-8") as trace_file,
        ):
            for q in questions:
                prediction, trace = predict(q, model, search_for(q["world_id"]))
                preds.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                trace_file.write(json.dumps(trace, ensure_ascii=False) + "\n")


def baseline_models(seed: int) -> list[Model]:
    """The floors: three fake models, the hand-written Evaluate rule, the first
    model that digs, and two hand-written Confronts (the naive vote, and the
    one that counts copies once)."""
    return [
        RandomModel(seed),
        AlwaysAbstainModel(),
        KeywordModel(),
        RuleEvaluateModel(),
        DigModel(),
        MajorityModel(),
        ChallengeModel(),
    ]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m docta_orchestrator.baseline")
    parser.add_argument("split", type=Path, help="generated split, e.g. data/test.jsonl")
    parser.add_argument("--out", type=Path, default=Path("predictions"))
    parser.add_argument("--traces", type=Path, default=Path("data/traces"))
    parser.add_argument("--seed", type=int, default=0, help="RandomModel seed")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL is not set (see .env.example)", file=sys.stderr)
        return 2
    try:
        questions = [r for _, r in read_records(args.split) if r["type"] == "question"]
        models = baseline_models(args.seed)
        with psycopg.connect(url, autocommit=True) as conn:
            run_models(
                questions, models, lambda wid: lambda q: search(conn, wid, q), args.out, args.traces
            )
    except (OSError, ValueError, psycopg.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{len(questions)} questions × {len(models)} models → {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
