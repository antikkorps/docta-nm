"""Prepares one collected domain for the loop:
`python -m docta_orchestrator.realtext.probe data/real/cities`.

Writes redactions.jsonl (the sentences cut so a third of the asked facts has
no answer) and questions.jsonl (built on the texts after the cuts, each
unanswerable one tagged natural or redacted, a third of every shape). Both are ground truth: for
loading and scoring, never a model input.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from docta_orchestrator.realtext.collect import write
from docta_orchestrator.realtext.questions import build
from docta_orchestrator.realtext.redact import apply, redact, tag


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path, help="a collected domain")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--share", type=float, default=1 / 3, help="asked facts left unanswered")
    args = parser.parse_args(argv)
    if not 0 < args.share < 1:
        parser.error("--share must lie strictly between 0 and 1")
    entities, articles = read(args.dir / "entities.jsonl"), read(args.dir / "articles.jsonl")
    cuts = redact(entities, articles, args.share, args.seed)
    texts = apply(articles, cuts)
    tagged = tag(build(entities, articles, args.seed), build(entities, texts, args.seed))
    questions = balanced(tagged, args.share, args.seed)
    write(args.dir / "redactions.jsonl", cuts)
    write(args.dir / "questions.jsonl", questions)
    report(questions, args.dir)
    return 0


def balanced(questions: list[dict], share: float, seed: int) -> list[dict]:
    """Every shape with `share` of its questions unanswerable, by dropping
    some of its majority side. A chain or a compare loses its answer when
    any of its facts is cut, so left alone they would be mostly
    unanswerable, and the shape alone would predict abstention. A shape
    with one side only is dropped for the same reason."""
    rng, kept = random.Random(seed), []
    for shape in sorted({q["shape"] for q in questions}):
        yes = [q for q in questions if q["shape"] == shape and q["answerable"]]
        no = [q for q in questions if q["shape"] == shape and not q["answerable"]]
        if not (yes and no):
            continue  # one side only: the shape alone would give the answer away
        want_no = half_up(len(yes) * share / (1 - share))
        if len(no) > want_no:
            no = rng.sample(no, want_no)
        else:
            yes = rng.sample(yes, min(len(yes), half_up(len(no) * (1 - share) / share)))
        kept += yes + no
    return sorted(kept, key=lambda q: q["question_id"])


def half_up(x: float) -> int:
    # round() goes to even: one answerable and two unanswerable would keep
    # no unanswerable question at all.
    return int(x + 0.5)


def report(questions: list[dict], where: Path) -> None:
    shapes = Counter(q["shape"] for q in questions)
    missing = Counter(q["unanswerable"] for q in questions if q["unanswerable"])
    print(f"{len(questions)} questions in {where}: {dict(shapes)}; unanswerable {dict(missing)}",
          file=sys.stderr)  # fmt: skip


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
