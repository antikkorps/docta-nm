"""`python -m docta_eval GOLD.jsonl PREDICTIONS.jsonl... [--json scores.json]`

Each predictions file is one model, named after the file.
"""

import argparse
import json
import sys
from pathlib import Path

from docta_eval.io import read_gold, read_predictions
from docta_eval.metrics import score
from docta_eval.report import render


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m docta_eval", description=__doc__)
    parser.add_argument("gold", type=Path, help="generated split, e.g. data/test.jsonl")
    parser.add_argument("predictions", type=Path, nargs="+", help="one JSONL file per model")
    parser.add_argument("--json", type=Path, help="also write the scores to this file")
    return parser.parse_args(argv)


def evaluate(gold: Path, predictions: list[Path]) -> dict[str, dict]:
    golds = read_gold(gold)
    keys = {(g.world_id, g.question_id) for g in golds}
    results: dict[str, dict] = {}
    for path in predictions:
        if path.stem in results:
            raise ValueError(f"model {path.stem!r} appears twice: rename one of its files")
        results[path.stem] = score(golds, read_predictions(path, keys))
    return results


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        results = evaluate(args.gold, args.predictions)
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
