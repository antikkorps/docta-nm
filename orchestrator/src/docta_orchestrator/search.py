"""The search tool exposed to models: full-text search over one world's sources.

A hit carries only what a reader would see. `fact_ids` is ground truth and
never leaves the database through this tool.
"""

import math
import re
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class Hit:
    source_id: str
    title: str
    body: str
    reliability: float


def words(text: str) -> list[str]:
    """The query's distinct words, lowercased, in order. Only letters and
    digits survive, so no tsquery operator can be injected."""
    return list(dict.fromkeys(re.findall(r"[^\W_]+", text.lower())))


def to_tsquery(text: str) -> str | None:
    """ORs the query's words: requiring all of them (plainto_tsquery) would
    almost never match a whole question."""
    return " | ".join(words(text)) or None


def idf(sources: int, containing: int) -> float:
    """BM25's inverse document frequency: a word in every source weighs
    almost nothing, a word in one weighs most. Always positive, so a match
    never lowers a score."""
    return math.log(1 + (sources - containing + 0.5) / (containing + 0.5))


def weights(conn: psycopg.Connection, world_id: str, terms: list[str]) -> list[float]:
    """Each term's idf within `world_id`: rarity is the world's own, so a
    word common in one corpus can still single out a source in another."""
    rows = conn.execute(
        "SELECT (SELECT count(*) FROM sources WHERE world_id = %s),"
        " array_agg((SELECT count(*) FROM sources"
        "  WHERE world_id = %s AND tsv @@ to_tsquery('simple', w)) ORDER BY i)"
        " FROM unnest(%s::text[]) WITH ORDINALITY AS u(w, i)",
        [world_id, world_id, terms],
        prepare=False,
    ).fetchone()
    total, counts = rows
    return [idf(total, c) for c in counts]


def search(conn: psycopg.Connection, world_id: str, query: str, k: int = 5) -> list[Hit]:
    """The `k` sources of `world_id` that best match `query`, best first: the
    rarer the query words a source holds, the higher it ranks. Unweighted,
    a long passage full of "in" and "was" beat the one naming the subject."""
    if k < 1:
        # LIMIT 0 would pass for "nothing found", a negative LIMIT fails in SQL.
        raise ValueError(f"k must be at least 1, got {k}")
    terms = words(query)
    if not terms:
        return []
    rows = conn.execute(
        "SELECT source_id, title, body, reliability"
        " FROM sources, to_tsquery('simple', %s) AS q"
        " WHERE world_id = %s AND tsv @@ q"
        " ORDER BY (SELECT sum(wt) FROM unnest(%s::text[], %s::float8[]) AS u(w, wt)"
        "  WHERE tsv @@ to_tsquery('simple', w)) DESC,"
        # ts_rank, then source_id, break ties, so results are deterministic.
        " ts_rank(tsv, q) DESC, source_id LIMIT %s",
        [" | ".join(terms), world_id, terms, weights(conn, world_id, terms), k],
        # Never prepared: a generic plan ignores world_id and scans the whole
        # full-text index, 20x slower than the per-world plan on 20k worlds.
        prepare=False,
    ).fetchall()
    return [Hit(*row) for row in rows]
