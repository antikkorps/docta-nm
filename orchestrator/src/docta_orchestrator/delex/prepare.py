"""Prepares one rewritten domain for loading:
`python -m docta_orchestrator.delex.prepare data/delex/bridges/rewritten`.

Writes cuts.jsonl (sentences cut so a third of the asked facts has no
answer) and questions.jsonl (built on the texts after the cuts, each
unanswerable one tagged natural or redacted, a third of every shape, so the
shape alone never predicts abstention). Both are ground truth: for training
targets and scoring, never a model's input."""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

from docta_orchestrator.delex.leaks import read
from docta_orchestrator.delex.questions import FACTS, Index, build, self_named, stating
from docta_orchestrator.realtext.collect import write
from docta_orchestrator.realtext.entities import SCHEMA
from docta_orchestrator.realtext.questions import named
from docta_orchestrator.realtext.redact import apply, tag

SHARE, SEED = 1 / 3, 42


def asked(entities: list[dict]) -> list[tuple[dict, str]]:
    return [(e, f) for e in entities for f, *_ in FACTS.get(e["kind"], [])
            if e["facts"].get(f) and not self_named(e, f)]  # fmt: skip


def redact(entities: list[dict], articles: list[dict], share: float, seed: int) -> list[dict]:
    """Cuts for enough stated facts that `share` of the asked facts have no
    answer, counting those the texts already leave out."""
    index, facts = Index(articles), asked(named(entities))
    said = [(e, f) for e, f in facts if stating(index, e, f)]
    need = round(share * len(facts)) - (len(facts) - len(said))
    chosen = random.Random(seed).sample(said, max(0, min(need, len(said))))
    out = []
    for e, f in sorted(chosen, key=lambda x: (int(x[0]["qid"][1:]), x[1])):
        cut = dict.fromkeys((q, s) for q, s, _ in stating(index, e, f))
        out += [{"schema": SCHEMA, "type": "redaction", "subject": e["qid"], "fact": f,
                 "article": q, "sentence": s} for q, s in cut]  # fmt: skip
    return out


def balanced(questions: list[dict], share: float, seed: int) -> list[dict]:
    """Every shape and fact with `share` of its questions unanswerable, by
    dropping some of its majority side; a group with one side only is
    dropped, since it alone would give the answer away. Per fact, not only
    per shape: texts rarely quote Wikidata's student counts, so "How many
    students" would otherwise mean "not enough". The probe balances the
    same way; this package may not import the probe's modules (the
    ground-truth guard), so the rule is written again here."""
    rng, kept = random.Random(seed), []
    for group in sorted({(q["shape"], q["fact"]) for q in questions}):
        mine = [q for q in questions if (q["shape"], q["fact"]) == group]
        yes = [q for q in mine if q["answerable"]]
        no = [q for q in mine if not q["answerable"]]
        if not (yes and no):
            continue
        want_no = int(len(yes) * share / (1 - share) + 0.5)
        if len(no) > want_no:
            no = rng.sample(no, want_no)
        else:
            yes = rng.sample(yes, min(len(yes), int(len(no) * (1 - share) / share + 0.5)))
        kept += yes + no
    return sorted(kept, key=lambda q: q["question_id"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path, help="a rewritten domain (delex.build --out)")
    args = parser.parse_args(argv)
    entities, articles = read(args.dir / "entities.jsonl"), read(args.dir / "articles.jsonl")
    cuts = redact(entities, articles, SHARE, SEED)
    texts = apply(articles, cuts)
    tagged = tag(build(entities, articles, SEED), build(entities, texts, SEED))
    questions = balanced(tagged, SHARE, SEED)
    write(args.dir / "cuts.jsonl", cuts)
    write(args.dir / "questions.jsonl", questions)
    shapes = Counter(q["shape"] for q in questions)
    missing = Counter(q["unanswerable"] for q in questions if q["unanswerable"])
    print(f"{len(questions)} questions: {dict(shapes)}; unanswerable {dict(missing)}",
          file=sys.stderr)  # fmt: skip


if __name__ == "__main__":
    main()
