"""Questions on the frozen real text, from the Wikidata answer key:
`python -m docta_orchestrator.realtext.questions data/real/cities`.

The wordings are the ones the cores learned on the invented worlds where a
real relation has one (a founding, a birth year), so a drop in score is a
drop in reading, not in parsing the question. Links the worlds never had (a
birthplace, a language's release, its designer) get new wordings, and each
question says which it uses. Whether a question is answerable is judged on
the texts (see stated.py), never assumed from the key.
"""

import argparse
import json
import random
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

from docta_orchestrator.models.templates import COMPARISONS, QUESTIONS
from docta_orchestrator.realtext.collect import write
from docta_orchestrator.realtext.entities import SCHEMA
from docta_orchestrator.realtext.stated import Index

# (kind, fact) → seen question predicate, or None for a new wording below.
ASKED = {("city", "inception"): "founded_in", ("person", "birth_year"): "born_in",
         ("language", "inception"): None}  # fmt: skip
RELEASED = "In what year was {s} first released?"
RELEASED_EARLIER = "Which was released earlier, {s} or {o}?"
# The first hop of a chain: the link, the entity it leads to, its wording.
CHAINS = {"born_in": "the city where {s} was born", "designed_by": "the designer of {s}"}
NOUN = {"inception": "founding or release year", "birth_year": "birth year"}


def build(entities: list[dict], articles: list[dict], seed: int = 0) -> list[dict]:
    """Every retrieve and chain question the key allows, and up to one
    comparison per entity of each kind, numbered in a fixed order."""
    rng, index = random.Random(seed), Index(articles)
    entities = named(entities)
    known = {e["qid"]: e for e in entities}
    out = [q for e in entities for q in retrieve(e, index, rng)]
    out += [q for e in entities for q in chains(e, known, index, rng)]
    for kind in ("city", "language"):
        out += compare([e for e in entities if e["kind"] == kind], index, rng)
    return [{"schema": SCHEMA, "type": "question", "question_id": f"q{i:05d}", **q}
            for i, q in enumerate(out)]  # fmt: skip


def named(entities: list[dict]) -> list[dict]:
    """Each entity with the name a question gives it: its label, or its full
    title when another entity shares the label (two Salems)."""
    count = Counter(e["label"] for e in entities)
    return [{**e, "name": e["title"] if count[e["label"]] > 1 else e["label"]} for e in entities]


def asked(e: dict) -> tuple[str, list[int]] | None:
    """The year fact a question asks about this entity, if it has one."""
    for (kind, fact), _ in ASKED.items():
        if e["kind"] == kind and e["facts"].get(fact):
            return fact, e["facts"][fact]
    return None


def wording(e: dict, subject: str, rng: random.Random) -> tuple[str, bool]:
    """The question for `e`'s year fact about `subject`, and whether seen."""
    pred = ASKED[(e["kind"], asked(e)[0])]
    if pred is None:
        return RELEASED.format(s=subject), False
    return rng.choice(QUESTIONS[pred]).format(s=subject), True


def retrieve(e: dict, index: Index, rng: random.Random) -> list[dict]:
    if asked(e) is None:
        return []
    fact, years = asked(e)
    body, seen = wording(e, e["name"], rng)
    stated = index.stated(e["qid"], e["name"], years)
    gap = f"no text states the {NOUN[fact]} of {e['name']}"
    return [question(body, 1, "retrieve", stated, years, [e["qid"]], seen, gap)]


def chains(e: dict, known: dict, index: Index, rng: random.Random) -> list[dict]:
    """A chain through a single link only: with two designers, "the
    designer of X" names no one in particular."""
    out = []
    for link, phrase in CHAINS.items():
        targets = [known[q] for q in e.get(link, []) if q in known]
        if len(targets) != 1 or asked(targets[0]) is None:
            continue
        target = targets[0]
        body, _ = wording(target, phrase.format(s=e["name"]), rng)
        hop = index.links((e["qid"], e["label"]), (target["qid"], target["label"]), link)
        stated = index.stated(target["qid"], target["name"], asked(target)[1]) if hop else None
        gap = f"no text ties {e['name']} to {target['name']} and states its year"
        out.append(question(body, 2, "chain", stated, asked(target)[1],
                            [e["qid"], target["qid"]], False, gap))  # fmt: skip
    return out


def compare(group: list[dict], index: Index, rng: random.Random) -> list[dict]:
    """As many pairs as entities, among those whose years cannot overlap
    whatever value is right: with 1749 against 1749 or 1750 no answer is
    fair."""
    pairs = [(a, b) for a, b in combinations(group, 2) if apart(a, b)]
    out = []
    for a, b in rng.sample(pairs, min(len(group), len(pairs))):
        first = (a if max(asked(a)[1]) < min(asked(b)[1]) else b)["name"]
        seen = a["kind"] == "city"
        template = COMPARISONS["founded_in"] if seen else RELEASED_EARLIER
        both = all(index.states(x["qid"], x["name"], asked(x)[1]) for x in (a, b))
        gap = f"no text states the year of both {a['name']} and {b['name']}"
        body = template.format(s=a["name"], o=b["name"])
        out.append(question(body, 1, "compare", first if both else None, [first],
                            [a["qid"], b["qid"]], seen, gap))  # fmt: skip
    return out


def apart(a: dict, b: dict) -> bool:
    ya, yb = asked(a), asked(b)
    return bool(ya and yb) and (max(ya[1]) < min(yb[1]) or max(yb[1]) < min(ya[1]))


def question(body, hops, primitive, stated, values, subjects, seen, gap) -> dict:
    """One record; `stated` is the answer the texts give, or None. An
    unanswerable question exports spot_missing, as in the worlds, and keeps
    the key's value in `missing` for the hand check."""
    values, answerable = [str(v) for v in values], stated is not None
    return {
        "body": body, "hops": hops, "primitive": primitive if answerable else "spot_missing",
        "answerable": answerable, "answer": stated,
        "values": values, "multi_valued": len(values) > 1, "subjects": subjects,
        "wording": "seen" if seen else "new",
        "unanswerable": None if answerable else "natural",
        "missing": None if answerable else f"{gap} (key: {', '.join(values)})",
    }  # fmt: skip


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dir", type=Path, help="a collected domain")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    read = [[json.loads(line) for line in (args.dir / f"{name}.jsonl").open(encoding="utf-8")]
            for name in ("entities", "articles")]  # fmt: skip
    out = build(*read, seed=args.seed)
    write(args.dir / "questions.jsonl", out)
    unanswerable = sum(not q["answerable"] for q in out)
    print(f"{len(out)} questions, {unanswerable} unanswerable, in {args.dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
