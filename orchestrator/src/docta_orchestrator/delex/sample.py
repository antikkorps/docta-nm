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
# Lead paragraphs of the best-known subjects per domain (lowest Wikidata
# ids), where a phrasing can name a subject without its name.
FAMOUS = 3


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
                                "title": b["title"], "path": t["path"], "qid": a["qid"],
                                "original": p, "rewritten": q, "famous": False})  # fmt: skip
    return out


def leads(pool: list[dict], k: int) -> list[dict]:
    """The first paragraph of the `k` articles of lowest id in `pool`."""
    first: dict[str, dict] = {}
    for p in pool:  # in id order already, paragraphs in reading order
        first.setdefault(p["qid"], p)
    return [p | {"famous": True} for p in list(first.values())[:k]]


def draw(domains: dict, n: int, seed: int, famous: int = 0) -> list[dict]:
    """The `famous` leads of each domain, then random paragraphs up to `n`,
    each domain's share in proportion to its articles."""
    rng = random.Random(seed)
    pools = {name: paragraphs(*domains[name], name) for name in sorted(domains)}
    picked = [p for name in pools for p in leads(pools[name], famous)]
    taken = {p["id"] for p in picked}
    for name, share in shares({k: len(v[0]) for k, v in domains.items()}, n - len(picked)).items():
        free = [p for p in pools[name] if p["id"] not in taken]
        picked += rng.sample(free, min(len(free), share))
    return picked


def shares(sizes: dict[str, int], n: int) -> dict[str, int]:
    """`n` split in proportion to `sizes`, largest remainders first, so the
    shares add up to `n` exactly (rounding each one can overshoot)."""
    total = sum(sizes.values())
    exact = {k: n * v / total for k, v in sorted(sizes.items())}
    out = {k: int(x) for k, x in exact.items()}
    for k in sorted(exact, key=lambda k: (out[k] - exact[k], k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("domains", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    domains = {d.name: (read(d / "articles.jsonl"), read(d / "rewritten/articles.jsonl"))
               for d in args.domains}  # fmt: skip
    rows = draw(domains, SIZE, SEED, FAMOUS)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} paragraphs → {args.out}")


if __name__ == "__main__":
    main()
