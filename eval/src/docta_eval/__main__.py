"""`python -m docta_eval GOLD.jsonl PREDICTIONS.jsonl... [--json scores.json]`

Each predictions file is one model, named after the file.
"""

import argparse
import json
import sys
from pathlib import Path

from docta_eval.io import read_gold, read_predictions, read_traces
from docta_eval.metrics import score
from docta_eval.process import process_scores
from docta_eval.report import render
from docta_eval.traps import trap_scores


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m docta_eval", description=__doc__)
    parser.add_argument("gold", type=Path, help="generated split, e.g. data/test.jsonl")
    parser.add_argument("predictions", type=Path, nargs="+", help="one JSONL file per model")
    parser.add_argument(
        "--traces", type=Path, help="directory of <model>.jsonl traces: also score the reflection"
    )
    parser.add_argument("--json", type=Path, help="also write the scores to this file")
    return parser.parse_args(argv)


def evaluate(gold: Path, predictions: list[Path], traces: Path | None = None) -> dict[str, dict]:
    golds = read_gold(gold)
    keys = {(g.world_id, g.question_id) for g in golds}
    results: dict[str, dict] = {}
    for path in predictions:
        if path.stem in results:
            raise ValueError(f"model {path.stem!r} appears twice: rename one of its files")
        preds = read_predictions(path, keys)
        results[path.stem] = score(golds, preds)
        traps = trap_scores(golds, preds)
        if traps:
            results[path.stem]["traps"] = traps
        if traces is not None:
            model_traces = read_traces(traces / f"{path.stem}.jsonl", keys)
            _check_agreement(preds, model_traces, path.stem)
            results[path.stem]["process"] = process_scores(golds, model_traces)
    return results


def _check_agreement(preds: dict, traces: dict, model: str) -> None:
    """A stale traces directory would mix two runs' numbers without a word."""
    for key, trace in traces.items():
        if key in preds and preds[key].abstained != trace["decision"]["abstained"]:
            raise ValueError(f"{model}: trace of {key[0]}/{key[1]} disagrees with its prediction")


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        results = evaluate(args.gold, args.predictions, args.traces)
        print(render(results))
        if args.json:
            payload = {"gold": str(args.gold), "models": results}
            args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
