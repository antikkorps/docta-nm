"""Scores of the human-correction protocol (orchestrator protocol.py).

Each answerable chain ran with no hint, the right bridge and a wrong one:
- accuracy per variant, and `recovered`: the share of chains failed alone that
  the right hint turned into a right answer (resuming well);
- `fooled`: the share of wrong-hint runs that answered what the wrong bridge
  would give (checking what one is told, or not);
- `heard`: the share of hinted runs that reached the hint (a model sure of its
  answer at round 1 never hears it), and `reoriented`: among those, the share
  whose next search went after the hinted name.

`python -m docta_eval.human data/test.jsonl predictions/test/human`
"""

import json
import sys
from pathlib import Path

from docta_eval.io import read_gold, read_predictions, read_traces
from docta_eval.metrics import Gold, Key, Pred, mentions, normalize, ratio
from docta_eval.report import cell

VARIANTS = ("none", "right", "wrong")
COLUMNS = [("none", "accuracy"), ("right", "accuracy"), ("wrong", "accuracy"),
           ("right", "heard"), ("right", "reoriented"), ("wrong", "reoriented")]  # fmt: skip


def right(gold: Gold, pred: Pred) -> bool:
    return (
        not pred.abstained
        and pred.answer is not None
        and (normalize(pred.answer) == normalize(gold.answer or ""))
    )


def reoriented(trace: dict, name: str) -> bool | None:
    """None when the hint never came; else whether a later plan sought `name`."""
    states = trace["states"]
    at = next((i for i, e in enumerate(states) if e["state"] == "human"), None)
    if at is None:
        return None
    later = (q for e in states[at:] if e["state"] == "plan" for q in e.get("queries", []))
    return any(mentions(q, name) for q in later)


def variant_scores(golds, preds, traces, hints, variant: str) -> dict:
    keys = list(hints)
    scores = {"accuracy": ratio(sum(right(golds[k], preds[k]) for k in keys), len(keys))}
    if variant != "none":
        moves = [reoriented(traces[k], hints[k][variant]) for k in keys]
        heard = [m for m in moves if m is not None]
        scores |= {
            "heard": ratio(len(heard), len(keys)),
            "reoriented": ratio(sum(heard), len(heard)),
        }
    return scores


def human_scores(golds: dict[Key, Gold], runs: dict, hints: dict[Key, dict]) -> dict:
    """runs: variant → (predictions, traces), each keyed like `hints`."""
    out = {v: variant_scores(golds, *runs[v], hints, v) for v in runs}
    (alone, _), (helped, _), (misled, _) = runs["none"], runs["right"], runs["wrong"]
    failed = [k for k in hints if not right(golds[k], alone[k])]
    baited = [k for k in hints if hints[k]["wrong_answer"] is not None]
    fooled = [k for k in baited if misled[k].answer is not None
              and normalize(misled[k].answer) == normalize(hints[k]["wrong_answer"])]  # fmt: skip
    out["recovered"] = ratio(sum(right(golds[k], helped[k]) for k in failed), len(failed))
    out["fooled"] = ratio(len(fooled), len(baited))
    return out


def load_runs(directory: Path, model: str, keys: set[Key]) -> dict:
    return {
        v: (
            read_predictions(directory / f"{model}.{v}.jsonl", keys),
            read_traces(directory / "traces" / f"{model}.{v}.jsonl", keys),
        )
        for v in VARIANTS
    }


def render(results: dict[str, dict]) -> str:
    heads = [f"{k[:6]} {v}" for v, k in COLUMNS] + ["recovered", "fooled"]
    width = max(len(m) for m in [*results, "model"])
    lines = [f"{'model':<{width}}  " + "  ".join(f"{h:>14}" for h in heads)]
    for model, s in results.items():
        cells = [cell(s[v][k]) for v, k in COLUMNS] + [cell(s["recovered"]), cell(s["fooled"])]
        lines.append(f"{model:<{width}}  " + "  ".join(f"{c:>14}" for c in cells))
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m docta_eval.human GOLD.jsonl PROTOCOL_DIR", file=sys.stderr)
        return 2
    gold_path, directory = Path(argv[0]), Path(argv[1])
    try:
        golds = {(g.world_id, g.question_id): g for g in read_gold(gold_path)}
        lines = (directory / "hints.jsonl").read_text(encoding="utf-8").splitlines()
        hints = {(h["world_id"], h["question_id"]): h for h in map(json.loads, lines)}
        models = sorted(p.name.removesuffix(".none.jsonl") for p in directory.glob("*.none.jsonl"))
        results = {
            m: human_scores(golds, load_runs(directory, m, set(hints)), hints) for m in models
        }
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(render(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
