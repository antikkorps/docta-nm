"""The search tool exposed to models: full-text search over one world's sources.

A hit carries only what a reader would see. `fact_ids` is ground truth and
never leaves the database through this tool.
"""

import re
from dataclasses import dataclass

import psycopg


@dataclass(frozen=True)
class Hit:
    source_id: str
    title: str
    body: str
    reliability: float


def to_tsquery(text: str) -> str | None:
    """ORs the query's words: requiring all of them (plainto_tsquery) would
    almost never match a whole question. Only letters and digits survive, so
    no tsquery operator can be injected."""
    words = dict.fromkeys(re.findall(r"[^\W_]+", text.lower()))
    return " | ".join(words) or None


def search(conn: psycopg.Connection, world_id: str, query: str, k: int = 5) -> list[Hit]:
    """The `k` sources of `world_id` that best match `query`, best first."""
    if k <= 0:
        # A negative LIMIT is a Postgres error and 0 silently returns nothing.
        raise ValueError(f"k must be positive, got {k}")
    tsquery = to_tsquery(query)
    if tsquery is None:
        return []
    rows = conn.execute(
        "SELECT source_id, title, body, reliability"
        " FROM sources, to_tsquery('simple', %s) AS q"
        " WHERE world_id = %s AND tsv @@ q"
        # source_id breaks rank ties, so results are deterministic.
        " ORDER BY ts_rank(tsv, q) DESC, source_id LIMIT %s",
        [tsquery, world_id, k],
        # Never prepared: a generic plan ignores world_id and scans the whole
        # full-text index, 20x slower than the per-world plan on 20k worlds.
        prepare=False,
    ).fetchall()
    return [Hit(*row) for row in rows]
