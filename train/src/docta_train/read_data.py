"""Builds the Read dataset: question + excerpts judged enough → the answer.

The Read core points at the answer in what was read; code copies the words
there. Its examples are the Evaluate examples labelled enough, joined to the
answer and the sources that state it (ground truth: training and scoring
only, never an input). Comparisons become one example per option, the option
named again after the question: the core reads each value, code compares.
A chain teaches its bridge: on what the first search read, the core points
at the name to search next (the city, the guild), in the source stating the
link; once the bridge's search read the second hop, at the answer.
Searching the bridge is code's job; no rule splits the question, so the
held-out link wordings of ood test whether the core reads which name the
chain needs.
Trap questions are left out: weighing contradicting sources is Confront's
step, not Read's.

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

Taught = tuple[str, str, tuple[str, ...], str]  # (question as read, answer, its sources, head)


@dataclass(frozen=True)
class ReadExample:
    world_id: str
    question_id: str
    split: str
    question: str  # as the core reads it: a comparison names its option last
    excerpts: tuple[Hit, ...]
    answer: str
    sources: tuple[str, ...]  # where the answer is stated
    # Which pointer learns it: "answer" (what answers the question) or
    # "bridge" (the name to search next). One pointer for both mixed them up.
    head: str = "answer"


def asked(question: dict, world: dict) -> list[tuple[str, str]]:
    """(question as read, answer) pairs: one, or one per compared option."""
    if question["primitive"] == "retrieve":
        return [(question["text"], question["answer"])]
    names = {e["id"]: e["name"] for e in world["entities"]}
    facts = {f["id"]: f for f in world["facts"]}
    compared = (facts[f] for f in question["required_facts"])
    return [(f"{question['text']} {names[f['subj']]}", str(f["obj"])) for f in compared]


def bridge(question: dict, world: dict) -> tuple[str, str]:
    """(bridge name, source stating the link): the first hop's object."""
    names = {e["id"]: e["name"] for e in world["entities"]}
    link = next(f for f in world["facts"] if f["id"] == question["required_facts"][0])
    return names[link["obj"]], question["required_sources"][0]


def chain_target(question: dict, world: dict, read: set[str]) -> list[Taught]:
    """The answer once both hops were read, the bridge while only the link
    was: whatever the core should look at next."""
    name, link = bridge(question, world)
    if set(question["required_sources"]) <= read:
        second = (question["required_sources"][1],)
        return [(question["text"], question["answer"], second, "answer")]
    return [(question["text"], name, (link,), "bridge")] if link in read else []


def taught(question: dict, world: dict, e: Example) -> list[Taught]:
    """(question as read, answer, sources stating it, head) the example teaches."""
    if question["trap"] or not question["answerable"]:
        return []
    if question["primitive"] == "chain":
        return chain_target(question, world, {h.source_id for h in e.excerpts})
    if not e.label or question["primitive"] not in ("retrieve", "compare"):
        return []
    sources = tuple(question["required_sources"])
    return [(text, answer, sources, "answer") for text, answer in asked(question, world)]


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
        for text, answer, sources, head in taught(q, worlds[e.world_id], e):
            yield ReadExample(
                e.world_id, e.question_id, e.split, text, e.excerpts, answer, sources, head
            )


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
            if "head" not in r:
                # Read back as answer targets, its bridges would mix the two pointers.
                raise ValueError(f"{path} predates the two pointers: rerun `just read-data`")
            if r["head"] not in ("answer", "bridge"):
                # Anything but "bridge" would train silently as an answer target.
                raise ValueError(f"{path}: unknown pointer head {r['head']!r}")
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
