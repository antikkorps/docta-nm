"""Loads one prepared domain of the probe into Postgres:
`python -m docta_orchestrator.realtext.load data/real/cities`.

The texts after the cuts become the sources of one world per domain
(`real_cities`), so a search looks across a whole corpus, as on the web. The
answer key goes to `facts` and the questions keep their answers: ground
truth, never returned by `search()`. articles.jsonl stays as collected.
"""

import os
import sys
from pathlib import Path

import psycopg

from docta_orchestrator.jsonl import WorldBatch
from docta_orchestrator.load import load_world
from docta_orchestrator.realtext.probe import read
from docta_orchestrator.realtext.redact import apply
from docta_orchestrator.realtext.stated import sentences

# Wikipedia gives no reliability signal: one value for all, said in the
# scores, so Confront weighs independent paragraphs by count alone.
RELIABILITY = 0.7
# The Read core reads 576 tokens, question and five excerpts included.
WORDS = 120


def chunks(paragraph: str, limit: int = WORDS) -> list[str]:
    """The paragraph in pieces of at most `limit` words, cut between
    sentences only: one sentence over the limit stays whole."""
    out: list[list[str]] = [[]]
    for sentence in sentences(paragraph):
        if out[-1] and len(" ".join([*out[-1], sentence]).split()) > limit:
            out.append([])
        out[-1].append(sentence)
    return [" ".join(part) for part in out if part and " ".join(part).strip()]


def sources(articles: list[dict]) -> list[dict]:
    out = []
    for a in articles:
        n = 0
        for section in a["sections"]:
            for paragraph in section["paragraphs"]:
                for text in chunks(paragraph):
                    title = " — ".join(section["path"])
                    out.append({"source_id": f"{a['qid']}:{n}", "title": title,
                                "reliability": RELIABILITY, "text": text, "facts": [],
                                "false_claims": [], "copy_of": None})  # fmt: skip
                    n += 1
    return out


def facts(entities: list[dict]) -> list[dict]:
    return [{"id": f"{e['qid']}:{name}:{i}", "subj": e["qid"], "pred": name, "obj": value}
            for e in entities for name, values in e["facts"].items()
            for i, value in enumerate(values)]  # fmt: skip


def question(q: dict) -> dict:
    # A paragraph has no fact ids: citations are scored as "supported" later.
    return {"question_id": q["question_id"], "text": q["body"], "primitive": q["primitive"],
            "hops": q["hops"], "answerable": q["answerable"], "answer": q["answer"],
            "missing": q["missing"], "required_facts": [], "required_sources": [],
            "trap": None, "expected_exit": None}  # fmt: skip


def world_of(domain: str) -> str:
    return f"real_{domain}"


def batch(domain: str, entities, articles, cuts, questions) -> WorldBatch:
    named = [{"id": e["qid"], "kind": e["kind"], "name": e["label"]} for e in entities]
    world = {"world_id": world_of(domain), "seed": 0, "split": "real",
             "entities": named, "facts": facts(entities)}  # fmt: skip
    return WorldBatch(world, sources(apply(articles, cuts)), [question(q) for q in questions])


def store(b: WorldBatch) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"]) as conn, conn.transaction():
        load_world(conn, b)


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print("usage: python -m docta_orchestrator.realtext.load DOMAIN_DIR", file=sys.stderr)
        return 2
    where = Path(argv[0])
    names = ("entities", "articles", "redactions", "questions")
    entities, articles, cuts, questions = (read(where / f"{n}.jsonl") for n in names)
    # The domain the collector wrote, not the folder's name: a copy elsewhere
    # must load as the same world.
    b = batch(entities[0]["domain"], entities, articles, cuts, questions)
    store(b)
    print(f"loaded {b.world['world_id']}: {len(questions)} questions", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
