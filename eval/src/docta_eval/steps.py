"""The method step by step, so a blockage is pinned to one step.

Each step of the method gets one row per measure, read from a model's
scores.json on test, val and ood side by side, and the change since the last
release's snapshot (scores/steps-<version>.json, kept in the repository so
anyone can follow each step from one version to the next).
"""

import argparse
import json
import re
import sys
from pathlib import Path

from docta_eval.report import cell

Path_ = tuple[str, ...]
# step → (main measure, supporting measure), each (name, where it is read in a
# model's scores; "human" paths are read from the human-correction protocol).
STEPS: list[tuple[str, tuple[str, Path_], tuple[str, Path_]]] = [
    ("search", ("reading_recall", ("process", "reading_recall")),
               ("reading_precision", ("process", "reading_precision"))),
    ("judge", ("abstention_f1", ("overall", "abstention_f1")),
              ("false_abstention_rate", ("overall", "false_abstention_rate"))),
    # retrieve only: a comparison's arithmetic is code's, not reading.
    ("read", ("retrieve_accuracy", ("by_primitive", "retrieve", "accuracy")),
             ("citation_precision", ("overall", "citation_precision"))),
    ("dig", ("chain_justified_accuracy", ("by_primitive", "chain", "justified_accuracy")),
            ("dig_recovery_rate", ("process", "dig_recovery_rate"))),
    ("confront", ("trap_accuracy", ("traps", "all", "accuracy")),
                 ("trap_fooled_rate", ("traps", "all", "fooled_rate"))),
    ("conclude", ("justified_accuracy", ("overall", "justified_accuracy")),
                 ("early_conclusion_rate", ("process", "early_conclusion_rate"))),
    ("human", ("recovered", ("human", "recovered")), ("fooled", ("human", "fooled"))),
]  # fmt: skip
Table = dict[str, dict[str, dict[str, float | None]]]  # step → measure → split → value


def pick(scores: dict, path: Path_) -> float | None:
    for key in path:
        if not isinstance(scores, dict) or key not in scores:
            return None
        scores = scores[key]
    return scores


def step_table(by_split: dict[str, dict | None], human: dict[str, dict] | None = None) -> Table:
    """A split never run (None) is left out: absent, never a zero. The human
    row covers only the splits the protocol ran on."""
    runs = {s: v for s, v in by_split.items() if v is not None}
    protocol = {s: {"human": h} for s, h in (human or {}).items() if s in runs}
    return {
        step: {
            name: {s: pick(v, path) for s, v in (protocol if path[0] == "human" else runs).items()}
            for name, path in measures
        }
        for step, *measures in STEPS
    }


def against(now: Table, then: Table) -> Table:
    """now − then per cell; None where either side is undefined."""
    return {
        step: {
            name: {s: change(v, then.get(step, {}).get(name, {}).get(s)) for s, v in splits.items()}
            for name, splits in measures.items()
        }
        for step, measures in now.items()
    }


def change(value: float | None, old: float | None) -> float | None:
    return None if value is None or old is None else value - old


def signed(value: float | None) -> str:
    return "" if value is None else f"{value:+.3f}"


def render(now: Table, moved: Table | None, splits: list[str]) -> str:
    heads = [*splits, *(f"Δ {s}" for s in splits if moved)]
    lines = [f"{'step':<9} {'measure':<25} " + " ".join(f"{h:>8}" for h in heads)]
    for step, measures in now.items():
        for name, values in measures.items():
            cells = [cell(values.get(s)) if s in values else "" for s in splits]
            if moved:
                cells += [signed(moved[step][name].get(s)) for s in splits]
            lines.append(f"{step:<9} {name:<25} " + " ".join(f"{c:>8}" for c in cells))
    return "\n".join(lines)


def load(predictions: Path, split: str, name: str, model: str) -> dict | None:
    path = predictions / split / name
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("models", data).get(model)


def version(text: str) -> tuple[int, ...]:
    if not re.fullmatch(r"\d+\.\d+\.\d+", text):
        raise ValueError(f"version {text!r} is not X.Y.Z")
    return tuple(map(int, text.split(".")))


def read_snapshot(path: Path) -> dict:
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        version(snapshot["version"])
        snapshot["models"]
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(f"{path}: not a step snapshot ({e})") from e
    return snapshot


def latest(snapshots: Path, model: str, before: str | None = None) -> dict | None:
    """The newest snapshot holding this model, older than the one being saved:
    a change is always measured forward."""
    found = [read_snapshot(p) for p in snapshots.glob("steps-*.json")]
    found = [s for s in found if model in s["models"]]
    if before:
        found = [s for s in found if version(s["version"]) < version(before)]
    return max(found, key=lambda s: version(s["version"]), default=None)


def save(snapshots: Path, number: str, note: str, model: str, table: Table) -> None:
    """One file per version, every model saved in it kept."""
    path = snapshots / f"steps-{number}.json"
    old = read_snapshot(path) if path.exists() else {"models": {}}
    # Saving a second model without a note keeps the checkpoints' note.
    note = note or old.get("note", "")
    snapshot = {"version": number, "note": note, "models": {**old["models"], model: table}}
    snapshots.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m docta_eval.steps", description=__doc__)
    p.add_argument("--model", default="core_read")
    p.add_argument("--predictions", type=Path, default=Path("predictions"))
    p.add_argument("--splits", nargs="+", help="default: test val ood, then each real_* scored")
    p.add_argument("--snapshots", type=Path, default=Path("scores"))
    p.add_argument("--save", metavar="VERSION", help="write scores/steps-VERSION.json")
    p.add_argument("--note", default="", help="saved with it: which checkpoints made it")
    return p.parse_args(argv)


def splits_of(a: argparse.Namespace) -> list[str]:
    """The worlds' splits, then each real-text domain already scored: a
    domain never run adds no empty column."""
    if a.splits:
        return a.splits
    real = sorted(p.parent.name for p in a.predictions.glob("real_*/scores.json"))
    return ["test", "val", "ood", *real]


def gather(a: argparse.Namespace) -> tuple[Table, dict | None]:
    """This model's table now, and the snapshot to measure the change from."""
    if a.save:
        version(a.save)  # refused before anything is read or written
    a.splits = splits_of(a)
    by_split = {s: load(a.predictions, s, "scores.json", a.model) for s in a.splits}
    if not any(by_split.values()):
        raise ValueError(f"no scores for {a.model} in {a.predictions}/{{{','.join(a.splits)}}}")
    human = {s: h for s in a.splits if (h := load(a.predictions, s, "human/scores.json", a.model))}
    return step_table(by_split, human), latest(a.snapshots, a.model, before=a.save)


def main(argv: list[str]) -> int:
    a = parse_args(argv)
    try:
        now, then = gather(a)
        if a.save:
            save(a.snapshots, a.save, a.note, a.model, now)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{a.model}" + (f" · since {then['version']}" if then else ""))
    print(render(now, against(now, then["models"][a.model]) if then else None, a.splits))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
