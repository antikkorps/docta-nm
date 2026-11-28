"""Collects one domain of the delexicalisation corpus:
`python -m docta_orchestrator.delex.collect bridges --out data/delex/bridges`.

Writes entities.jsonl (Wikidata values, aliases: what gets replaced and the
labels' source) and articles.jsonl (the text, pinned to its revision, with
its internal links). Raw, never trained on as is: the delexicaliser turns
it into training text. Rerunning reuses the cache."""

import argparse
import sys
from pathlib import Path

from docta_orchestrator.delex import domains, links
from docta_orchestrator.realtext import wiki
from docta_orchestrator.realtext.collect import SPARQL, write
from docta_orchestrator.realtext.fetch import Fetcher, FetchError
from docta_orchestrator.realtext.wikidata import bindings

# Enough text to learn the language from; small enough to collect politely
# (two requests per article, one second apart: about an hour per domain).
LIMIT = 1000


def articles(fetch, entities: list[dict]):
    for entity in entities:
        record = wiki.article_record(entity, fetch.json(wiki.API, wiki.params(entity["title"])))
        if record is None:
            print(f"no article for {entity['qid']} ({entity['title']})", file=sys.stderr)
            continue
        yield record | {"links": links.of(fetch, entity["title"])}


def asker(fetch):
    return lambda query: bindings(fetch.json(SPARQL, {"query": query, "format": "json"}))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("domain", choices=sorted(domains.DOMAINS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/delex/.cache"))
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args(argv)
    fetch = Fetcher(args.cache)
    try:
        entities = domains.collect(domains.DOMAINS[args.domain], asker(fetch), args.limit)
        args.out.mkdir(parents=True, exist_ok=True)
        write(args.out / "entities.jsonl", entities)
        n = write(args.out / "articles.jsonl", articles(fetch, entities))
    except FetchError as e:
        sys.exit(f"error: {e}: rerun to resume, answers already received are cached")
    print(f"{args.domain}: {len(entities)} entities, {n} articles in {args.out}")


if __name__ == "__main__":
    main()
