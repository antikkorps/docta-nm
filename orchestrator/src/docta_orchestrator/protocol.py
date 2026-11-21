"""The human-correction protocol: does the core resume well when a person
steps in mid-way, and does it still check what it is told?

Each answerable chain runs three times: with no hint, with the right bridge
("Orlan Teste lives in Varnelle.") and with a wrong one of the same kind. The
hints come from ground truth: this harness plays the human, so it may read
the world; models only hear the hint, through the human channel (human.py).

`python -m docta_orchestrator.protocol data/test.jsonl --out predictions/test/human`
"""

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import psycopg

from docta_orchestrator.baseline import predict
from docta_orchestrator.human import human_hint
from docta_orchestrator.jsonl import read_records
from docta_orchestrator.loop import Search
from docta_orchestrator.models import DigModel
from docta_orchestrator.models.templates import SENTENCES
from docta_orchestrator.search import Hit, search
from docta_orchestrator.states import Model

VARIANTS = ("none", "right", "wrong")


@dataclass(frozen=True)
class Hints:
    right: Hit
    wrong: Hit
    right_bridge: str
    wrong_bridge: str
    wrong_answer: str | None  # what a model fooled by the wrong hint would say


def hints(question: dict, world: dict, stated: set[str] | None = None) -> Hints | None:
    """The right and a wrong bridge for an answerable chain; None otherwise.
    `stated`: the fact ids the world's sources state (all, when None)."""
    if question["primitive"] != "chain" or not question["answerable"]:
        return None
    facts = {f["id"]: f for f in world["facts"]}
    link, outer = (facts[f] for f in question["required_facts"])
    names = {e["id"]: e["name"] for e in world["entities"]}
    wrong, bait = decoy(link["obj"], outer["pred"], world, stated)
    said = SENTENCES[link["pred"]][0]  # a seen sentence: the hint reads plainly
    subject = names[link["subj"]]
    return Hints(
        human_hint(said.format(s=subject, o=names[link["obj"]])),
        human_hint(said.format(s=subject, o=names[wrong])),
        names[link["obj"]],
        names[wrong],
        None if bait is None else str(names.get(bait["obj"], bait["obj"])),
    )


def decoy(bridge: str, pred: str, world: dict, stated: set[str] | None) -> tuple[str, dict | None]:
    """Another entity of the bridge's kind, and its `pred` fact. One whose fact
    a source states comes first: only then could obeying the hint give a wrong
    answer to measure."""
    kind = next(e["kind"] for e in world["entities"] if e["id"] == bridge)
    others = sorted(e["id"] for e in world["entities"] if e["kind"] == kind and e["id"] != bridge)
    fact = {f["subj"]: f for f in world["facts"] if f["pred"] == pred}
    readable = [o for o in others if o in fact and (stated is None or fact[o]["id"] in stated)]
    chosen = (readable or others)[0]
    return chosen, fact.get(chosen) if chosen in readable else None


def run_protocol(
    questions: list[dict],
    worlds: dict[str, dict],
    models: list[Model],
    search_for: Callable[[str], Search],
    out: Path,
    stated: dict[str, set[str]] | None = None,
) -> int:
    """Writes out/<model>.<variant>.jsonl, their traces and out/hints.jsonl."""
    told = stated or {}
    chains = [(q, h) for q in questions
              if (h := hints(q, worlds[q["world_id"]], told.get(q["world_id"])))]  # fmt: skip
    (out / "traces").mkdir(parents=True, exist_ok=True)
    with (out / "hints.jsonl").open("w", encoding="utf-8") as told:
        for q, h in chains:
            told.write(json.dumps({"world_id": q["world_id"], "question_id": q["question_id"],
                                   "right": h.right_bridge, "wrong": h.wrong_bridge,
                                   "wrong_answer": h.wrong_answer}) + "\n")  # fmt: skip
    for model in models:
        for variant in VARIANTS:
            write_variant(chains, model, variant, search_for, out)
    return len(chains)


def write_variant(chains, model, variant: str, search_for, out: Path) -> None:
    name = f"{model.name}.{variant}.jsonl"
    with (
        (out / name).open("w", encoding="utf-8") as preds,
        (out / "traces" / name).open("w", encoding="utf-8") as traces,
    ):
        for q, h in chains:
            hint = {"none": None, "right": h.right, "wrong": h.wrong}[variant]
            prediction, trace = predict(q, model, search_for(q["world_id"]), hint=hint)
            preds.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            traces.write(json.dumps(trace, ensure_ascii=False) + "\n")


def run_on_split(split: Path, models: list[Model], out: Path) -> int:
    """The protocol over a generated split, searching the loaded sources."""
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is not set (see .env.example)")
    records = [r for _, r in read_records(split)]
    worlds = {r["world_id"]: r for r in records if r["type"] == "world"}
    questions = [r for r in records if r["type"] == "question"]
    stated: dict[str, set[str]] = {}
    for r in records:
        if r["type"] == "source":
            stated.setdefault(r["world_id"], set()).update(r["facts"])
    with psycopg.connect(url, autocommit=True) as conn:
        search_for = lambda wid: lambda q: search(conn, wid, q)  # noqa: E731
        return run_protocol(questions, worlds, models, search_for, out, stated)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="python -m docta_orchestrator.protocol")
    p.add_argument("split", type=Path, help="generated split, e.g. data/test.jsonl")
    p.add_argument("--out", type=Path, default=Path("predictions/human"))
    args = p.parse_args(argv)
    try:
        n = run_on_split(args.split, [DigModel()], args.out)
    except (OSError, ValueError, psycopg.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"{n} chains × 3 runs → {args.out}/", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
