"""The sample checked by hand before any real-text score is believed:
`python -m docta_orchestrator.realtext.sample data/real/cities
data/real/languages --out data/real/sample.jsonl`.

So many questions of each kind (answerable, naturally absent, redacted),
drawn with a seed so the label error found can be reported and redone. Each
comes with what shows its label right or wrong: the sentences that state its
year, a chain's link, the cuts, and for an absence the years its article
does hold. Drawn from all domains pooled, so each domain's share follows its
size. Ground truth throughout: for the person checking, never a model.
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
# Any number of three or four digits: an absence is checked against every
# year its article could hold, not a handful.
_NUMBER = re.compile(r"\b\d{3,4}\b")


def sample(domains: list[Path], per_kind: int = 50, seed: int = 0) -> list[dict]:
    pool = [item for where in sorted(domains) for item in items(where)]
    rng, out = random.Random(seed), []
    for kind in KINDS:
        mine = [i for i in pool if i["kind"] == kind]
        out += sorted(rng.sample(mine, min(per_kind, len(mine))),
                      key=lambda i: (i["domain"], i["question_id"]))  # fmt: skip
    return out


def items(where: Path) -> list[dict]:
    entities = {e["qid"]: e for e in named(read(where / "entities.jsonl"))}
    articles = read(where / "articles.jsonl")
    texts = (Index(articles), Index(apply(articles, read(where / "redactions.jsonl"))))
    urls = {a["qid"]: a.get("url") for a in articles}
    return [item(q, entities, texts, urls) for q in read(where / "questions.jsonl")]


def item(q: dict, entities: dict, texts: tuple[Index, Index], urls: dict) -> dict:
    before, after = (evidence_of(q, entities, index) for index in texts)
    holders = q["subjects"] if q["shape"] == "compare" else q["subjects"][-1:]
    shown = {
        "states": [[a, s] for a, s, why in after if why == "year"],
        "link": [[a, s] for a, s, why in after if why == "link"],
        # What the cuts took from this question, not every cut on its subjects.
        "cut": [list(x) for x in before if x not in after],
        "years_nearby": [] if q["answerable"] else nearby(holders, texts[1]),
    }
    quoted = {a for rows in shown.values() for a, *_ in rows} | set(q["subjects"])
    shown["urls"] = {a: urls.get(a) for a in sorted(quoted)}
    return {"domain": entities[q["subjects"][0]]["domain"], "question_id": q["question_id"],
            "body": q["body"], "shape": q["shape"], "kind": q["unanswerable"] or "answerable",
            "answer": q["answer"], "values": q["values"], "subjects": q["subjects"],
            "evidence": shown, "verdict": None}  # fmt: skip


def evidence_of(q: dict, entities: dict, index: Index) -> list[tuple[str, str, str]]:
    """The sentences a label rests on in these texts: each year-holder's
    stated year, and a chain's link."""
    holders = q["subjects"] if q["shape"] == "compare" else q["subjects"][-1:]
    out = [(a, s, "year") for h in holders for a, s in years_of(entities[h], index)]
    out += [(a, s, "link") for a, s in link_of(q, entities, index)]
    return list(dict.fromkeys(out))


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
    return [[q, s] for q in holders for s in index.own.get(q, []) if _NUMBER.search(s)]


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
