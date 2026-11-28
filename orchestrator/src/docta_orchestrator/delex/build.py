"""Rewrites one collected domain into training text:
`python -m docta_orchestrator.delex.build data/delex/bridges --out
data/delex/bridges/rewritten --names data/delex/names.txt`.

Writes articles.jsonl and entities.jsonl in the probe's format, so its
question machinery runs on them, and mapping.json (each original name's
replacement, each article's shift and factor) apart: ground truth for the
leak check, never a model's input."""

import argparse
import json
import random
from pathlib import Path

from docta_orchestrator.delex import rewrite, words
from docta_orchestrator.realtext.collect import write
from docta_orchestrator.realtext.probe import read

# Fixed, so the same corpus always gives the same text.
SEED = 42


def rewrite_domain(articles: list[dict], entities: list[dict], pool: list[str], seed: int):
    names = words.Namer(pool, words.lowercase_words(articles)).assign(
        words.entity_words(articles, entities)
    )
    by_qid = {e["qid"]: e for e in entities}
    out_a, out_e, draws = [], [], {}
    for art in sorted(articles, key=lambda a: int(a["qid"][1:])):
        # One generator per article, seeded by its id: an article rewrites
        # the same way whatever else is in the corpus.
        shift, factor = rewrite.draw(random.Random(f"{seed}:{art['qid']}"))
        out_a.append(rewrite.article(art, names, shift, factor))
        out_e.append(rewrite.entity(by_qid[art["qid"]], names, shift, factor))
        draws[art["qid"]] = [shift, factor]
    return out_a, out_e, {"names": names, "draws": draws}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("raw", type=Path, help="a collected domain (entities, articles)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True, help="worldgen --emit-names")
    args = parser.parse_args(argv)
    articles, entities = read(args.raw / "articles.jsonl"), read(args.raw / "entities.jsonl")
    pool = args.names.read_text(encoding="utf-8").split()
    arts, ents, mapping = rewrite_domain(articles, entities, pool, SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "articles.jsonl", arts)
    write(args.out / "entities.jsonl", ents)
    text = json.dumps(mapping, ensure_ascii=False, sort_keys=True, indent=1)
    (args.out / "mapping.json").write_text(text + "\n", encoding="utf-8")
    print(f"{len(arts)} articles rewritten, {len(mapping['names'])} names replaced")


if __name__ == "__main__":
    main()
