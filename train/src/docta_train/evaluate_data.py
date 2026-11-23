"""Builds the Evaluate dataset: question + excerpts read → enough or not.

Each example is exactly what the core sees at inference: the question, and the
excerpts the real search tool returns for it. The label comes from ground
truth: enough means the question has an answer AND every source it needs was
read. So the core learns "is what I read sufficient?", not "does an answer
exist?".

`python -m docta_train.evaluate_data data/train.jsonl data/val.jsonl --out data/evaluate`
"""

import argparse
import json
import os
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import psycopg
from docta_orchestrator.jsonl import read_records
from docta_orchestrator.loop import Search
from docta_orchestrator.models.dig import revealed
from docta_orchestrator.models.keyword import compared
from docta_orchestrator.search import Hit, search

from docta_train.chain_truth import Bridge, Key, chain_bridges


@dataclass(frozen=True)
class Example:
    world_id: str
    question_id: str
    split: str
    question: str
    excerpts: tuple[Hit, ...]
    label: bool  # True: the excerpts are enough to answer


def enough(question: dict, hits: Iterable[Hit]) -> bool:
    read = {h.source_id for h in hits}
    return question["answerable"] and set(question["required_sources"]) <= read


def build_examples(
    questions: Iterable[dict],
    search_for: Callable[[str], Search],
    split: str,
    bridges: Mapping[Key, Bridge] | None = None,
) -> Iterator[Example]:
    """One example per question, searched the way the models plan: a single
    query made of the question itself. A question not yet enough gets a
    second one, as core_bridge sees it in round 2: what it holds once a name
    it brought in was searched (see dug_name)."""
    for q in questions:
        search = search_for(q["world_id"])
        hits = tuple(search(q["text"]))
        yield Example(q["world_id"], q["question_id"], split, q["text"], hits, enough(q, hits))
        name = None if enough(q, hits) else dug_name(q, hits, bridges or {})
        if name:
            read = {h.source_id for h in hits}
            dug = hits + tuple(h for h in search(name) if h.source_id not in read)
            yield Example(q["world_id"], q["question_id"], split, q["text"], dug, enough(q, dug))


def dug_name(question: dict, hits: tuple[Hit, ...], bridges: Mapping[Key, Bridge]) -> str | None:
    """The name core_bridge would search next: a chain's bridge once its link
    was read (where the Read core learns to point); otherwise, as a stand-in
    for wherever the core points, the first name a sentence about the
    subject brings in. Comparisons never dig. Evaluate thus sees more read
    than needed on every shape, not only on chains that have an answer."""
    if compared(question["text"]):
        return None
    bridge = bridges.get((question["world_id"], question["question_id"]))
    if bridge and bridge[1] in {h.source_id for h in hits}:
        return bridge[0]
    names = revealed(question["text"], hits)
    return names[0] if names else None


def write_examples(examples: Iterable[Example], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as out:
        for e in examples:
            out.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_examples(path: Path) -> Iterator[Example]:
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            r = json.loads(line)
            hits = tuple(Hit(**h) for h in r["excerpts"])
            yield Example(
                r["world_id"], r["question_id"], r["split"], r["question"], hits, r["label"]
            )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m docta_train.evaluate_data")
    parser.add_argument(
        "splits", type=Path, nargs="+", help="generated splits, e.g. data/train.jsonl"
    )
    parser.add_argument("--out", type=Path, default=Path("data/evaluate"))
    args = parser.parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL is not set (see .env.example)", file=sys.stderr)
        return 2
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            for path in args.splits:
                bridges = chain_bridges(r for _, r in read_records(path))
                questions = (r for _, r in read_records(path) if r["type"] == "question")
                examples = build_examples(
                    questions, lambda wid: lambda q: search(conn, wid, q), path.stem, bridges
                )
                n = write_examples(examples, args.out / f"{path.stem}.jsonl")
                print(f"{path.stem}: {n} examples", file=sys.stderr)
    except (OSError, ValueError, psycopg.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
