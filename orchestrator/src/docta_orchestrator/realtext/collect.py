"""Collects one domain of the real-text probe:
`python -m docta_orchestrator.realtext.collect cities --out data/real/cities`.

Writes entities.jsonl (the answer key, from Wikidata: scoring only) and
articles.jsonl (what a model may read, from Wikipedia) apart, so the key can
never slip into a model's input. Rerunning reuses the cache."""

import argparse
import json
import sys
from pathlib import Path

from docta_orchestrator.realtext import cities, languages, wiki
from docta_orchestrator.realtext.fetch import Fetcher, FetchError
from docta_orchestrator.realtext.wikidata import bindings

SPARQL = "https://query.wikidata.org/sparql"
DOMAINS = {"cities": cities.collect, "languages": languages.collect}
LIMITS = {"cities": 200, "languages": 150}


def write(path: Path, records) -> int:
    with path.open("w", encoding="utf-8") as f:
        count = sum(1 for r in records if f.write(json.dumps(r, ensure_ascii=False) + "\n"))
    return count


def articles(fetch: Fetcher, entities: list[dict]):
    for entity in entities:
        record = wiki.article_record(entity, fetch.json(wiki.API, wiki.params(entity["title"])))
        if record is None:
            print(f"no article for {entity['qid']} ({entity['title']})", file=sys.stderr)
            continue
        yield record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("domain", choices=sorted(DOMAINS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/real/.cache"))
    parser.add_argument("--limit", type=int, help="entities chosen (default per domain)")
    args = parser.parse_args(argv)
    try:
        run(args)
    except FetchError as e:
        sys.exit(f"error: {e}: rerun to resume, answers already received are cached")


def run(args: argparse.Namespace) -> None:
    fetch = Fetcher(args.cache)
    entities = DOMAINS[args.domain](
        lambda query: bindings(fetch.json(SPARQL, {"query": query, "format": "json"})),
        args.limit or LIMITS[args.domain],
    )
    args.out.mkdir(parents=True, exist_ok=True)
    write(args.out / "entities.jsonl", entities)
    n = write(args.out / "articles.jsonl", articles(fetch, entities))
    print(f"{args.domain}: {len(entities)} entities, {n} articles in {args.out}")


if __name__ == "__main__":
    main()
