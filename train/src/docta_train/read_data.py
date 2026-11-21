"""Builds the Read dataset: question + excerpts judged enough → the answer.

The Read core points at the answer in what was read; code copies the words
there. Its examples are the Evaluate examples labelled enough, joined to the
answer and the sources that state it (ground truth: training and scoring
only, never an input). Comparisons become one example per option, the option
named again after the question: the core reads each value, code compares.
Chains are left out: the orchestrator splits them into one-hop questions,
which retrieve examples already teach. Trap questions are left out too:
weighing contradicting sources is Confront's step, not Read's.

`python -m docta_train.read_data data/train.jsonl data/val.jsonl --out data/read`
"""

import argparse
import json
import sys
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from docta_orchestrator.jsonl import read_records
from docta_orchestrator.search import Hit

from docta_train.evaluate_data import Example, read_examples


@dataclass(frozen=True)
class ReadExample:
    world_id: str
    question_id: str
    split: str
    question: str  # as the core reads it: a comparison names its option last
    excerpts: tuple[Hit, ...]
    answer: str
    sources: tuple[str, ...]  # where the answer is stated


def asked(question: dict, world: dict) -> list[tuple[str, str]]:
    """(question as read, answer) pairs: one, or one per compared option."""
    if question["primitive"] == "retrieve":
        return [(question["text"], question["answer"])]
    names = {e["id"]: e["name"] for e in world["entities"]}
    facts = {f["id"]: f for f in world["facts"]}
    compared = (facts[f] for f in question["required_facts"])
    return [(f"{question['text']} {names[f['subj']]}", str(f["obj"])) for f in compared]


def build_read_examples(
    records: Iterable[dict], evaluated: Iterable[Example]
) -> Iterator[ReadExample]:
    worlds, questions = {}, {}
    for r in records:
        if r["type"] == "world":
            worlds[r["world_id"]] = r
        elif r["type"] == "question":
            questions[(r["world_id"], r["question_id"])] = r
    for e in evaluated:
        q = questions[(e.world_id, e.question_id)]
        if not e.label or q["trap"] or q["primitive"] not in ("retrieve", "compare"):
            continue
        sources = tuple(q["required_sources"])
        for text, answer in asked(q, worlds[e.world_id]):
            yield ReadExample(e.world_id, e.question_id, e.split, text, e.excerpts, answer, sources)


def write_reads(examples: Iterable[ReadExample], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for e in examples:
            out.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_reads(path: Path) -> Iterator[ReadExample]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            r = json.loads(line)
            hits = tuple(Hit(**h) for h in r.pop("excerpts"))
            yield ReadExample(**{**r, "excerpts": hits, "sources": tuple(r["sources"])})


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m docta_train.read_data")
    parser.add_argument("splits", type=Path, nargs="+", help="e.g. data/train.jsonl")
    parser.add_argument("--evaluate", type=Path, default=Path("data/evaluate"))
    parser.add_argument("--out", type=Path, default=Path("data/read"))
    args = parser.parse_args(argv)
    try:
        for path in args.splits:
            records = (r for _, r in read_records(path))
            evaluated = read_examples(args.evaluate / f"{path.stem}.jsonl")
            n = write_reads(
                build_read_examples(records, evaluated), args.out / f"{path.stem}.jsonl"
            )
            print(f"{path.stem}: {n} examples", file=sys.stderr)
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
