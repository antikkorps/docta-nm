"""The hand-check sample of step 3: paragraphs drawn at random, each beside
its rewriting, for a person to spot what the automatic check cannot (a name
made of common words, a fact carried by the phrasing).

`python -m docta_orchestrator.delex.sample data/delex/mountains
data/delex/bridges --out data/delex/sample.jsonl`"""

import argparse
import json
import random
from pathlib import Path

from docta_orchestrator.delex.leaks import read

# Short paragraphs (captions, one-liners) say little about a leak.
MIN_CHARS = 200
SIZE, SEED = 50, 0


def paragraphs(raw: list[dict], new: list[dict], domain: str) -> list[dict]:
    """Original and rewritten paragraphs side by side: the rewriting keeps
    every article's sections and paragraphs in place, so they align."""
    by_qid = {a["qid"]: a for a in new}
    out = []
    for a in sorted(raw, key=lambda a: int(a["qid"][1:])):
        b = by_qid.get(a["qid"])
        for i, (s, t) in enumerate(zip(a["sections"], b["sections"] if b else [], strict=False)):
            for j, (p, q) in enumerate(zip(s["paragraphs"], t["paragraphs"], strict=True)):
                if len(p) > MIN_CHARS:
                    out.append({"id": f"{domain}-{a['qid']}-{i}-{j}", "domain": domain,
                                "title": b["title"], "path": t["path"],
                                "original": p, "rewritten": q})  # fmt: skip
    return out


def draw(domains: dict, n: int, seed: int) -> list[dict]:
    """`n` paragraphs, each domain's share in proportion to its articles."""
    rng = random.Random(seed)
    total = sum(len(raw) for raw, _ in domains.values())
    picked = []
    for name in sorted(domains):
        raw, new = domains[name]
        pool = paragraphs(raw, new, name)
        picked += rng.sample(pool, min(len(pool), round(n * len(raw) / total)))
    return picked


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("domains", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    domains = {d.name: (read(d / "articles.jsonl"), read(d / "rewritten/articles.jsonl"))
               for d in args.domains}  # fmt: skip
    rows = draw(domains, SIZE, SEED)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} paragraphs → {args.out}")


if __name__ == "__main__":
    main()
