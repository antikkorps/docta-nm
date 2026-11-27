"""The sample checked by hand before any real-text score is believed:
`python -m docta_orchestrator.realtext.sample data/real/cities
data/real/languages --out data/real/sample.jsonl`.

So many questions of each kind (answerable, naturally absent, redacted),
drawn with a seed so the label error found can be reported and redone. Each
comes with what shows its label right or wrong: the sentences that state its
year, a chain's link, the cuts, and for an absence the years its article
does hold. Ground truth throughout: for the person checking, never a model.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

from docta_orchestrator.realtext.probe import read
from docta_orchestrator.realtext.questions import CHAINS, asked, named
from docta_orchestrator.realtext.redact import apply
from docta_orchestrator.realtext.stated import Index

KINDS = ("answerable", "natural", "redacted")
_YEAR = re.compile(r"\b1\d{3}\b|\b20\d{2}\b")
NEARBY = 6  # enough to see the year is not there, few enough to read


def sample(domains: list[Path], per_kind: int = 50, seed: int = 0) -> list[dict]:
    pool = [item for where in domains for item in items(where)]
    rng, out = random.Random(seed), []
    for kind in KINDS:
        mine = [i for i in pool if i["kind"] == kind]
        out += sorted(rng.sample(mine, min(per_kind, len(mine))),
                      key=lambda i: (i["domain"], i["question_id"]))  # fmt: skip
    return out


def items(where: Path) -> list[dict]:
    entities = {e["qid"]: e for e in named(read(where / "entities.jsonl"))}
    articles = read(where / "articles.jsonl")
    cuts = read(where / "redactions.jsonl")
    index = Index(apply(articles, cuts))
    urls = {a["qid"]: a.get("url") for a in articles}
    return [item(q, entities, index, cuts, urls) for q in read(where / "questions.jsonl")]


def item(q: dict, entities: dict, index: Index, cuts: list[dict], urls: dict) -> dict:
    holders = q["subjects"] if q["shape"] == "compare" else q["subjects"][-1:]
    years = [y for s in holders for y in years_of(entities[s], index)]
    evidence = {
        "states": [list(x) for x in years],
        "link": [list(x) for x in link_of(q, entities, index)],
        "cut": [[c["article"], c["sentence"]] for c in cuts if c["subject"] in q["subjects"]],
        "years_nearby": [] if q["answerable"] else nearby(holders, index),
        "urls": [urls.get(s) for s in q["subjects"]],
    }
    return {"domain": entities[q["subjects"][0]]["domain"], "question_id": q["question_id"],
            "body": q["body"], "shape": q["shape"], "kind": q["unanswerable"] or "answerable",
            "answer": q["answer"], "values": q["values"], "subjects": q["subjects"],
            "evidence": evidence, "verdict": None}  # fmt: skip


def years_of(e: dict, index: Index) -> list[tuple[str, str]]:
    fact = asked(e)
    return index.stating(e["qid"], e["name"], fact[1]) if fact else []


def link_of(q: dict, entities: dict, index: Index) -> list[tuple[str, str]]:
    if q["shape"] != "chain":
        return []
    a, b = (entities[s] for s in q["subjects"])
    link = next(k for k in CHAINS if b["qid"] in a.get(k, []))
    return index.linking((a["qid"], a["label"]), (b["qid"], b["label"]), link)


def nearby(holders: list[str], index: Index) -> list[list[str]]:
    found = [[q, s] for q in holders for s in index.own.get(q, []) if _YEAR.search(s)]
    return found[:NEARBY]


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("domains", type=Path, nargs="+", help="prepared domains (realtext.probe)")
    p.add_argument("--per-kind", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    drawn = sample(args.domains, args.per_kind, args.seed)
    args.out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in drawn))
    print(f"{len(drawn)} questions to check → {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
