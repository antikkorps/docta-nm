"""Loads JSONL v2 files into Postgres: `python -m docta_orchestrator.load data/*.jsonl`.

Postgres is a projection of the canonical JSONL: loading a world replaces it, so
reloading after `just gen` is always safe.
"""

import argparse
import os
import sys
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from docta_orchestrator.jsonl import WorldBatch, group_worlds, read_records


def load_world(conn: psycopg.Connection, batch: WorldBatch) -> None:
    """Replaces one world. Does not commit: the caller owns the transaction."""
    w = batch.world
    wid = w["world_id"]
    # ON DELETE CASCADE removes the world's entities, facts, sources and questions.
    conn.execute("DELETE FROM worlds WHERE world_id = %s", [wid])
    conn.execute(
        "INSERT INTO worlds (world_id, seed, split) VALUES (%s, %s, %s)",
        [wid, w["seed"], w["split"]],
    )
    with conn.cursor() as cur:
        for table, columns, rows in _rows(wid, batch):
            _copy(cur, table, columns, rows)


def _rows(wid: str, batch: WorldBatch) -> list[tuple[str, str, list[tuple]]]:
    """Each child table with its columns and this world's rows."""
    w = batch.world
    return [
        (
            "entities",
            "world_id, entity_id, kind, name",
            [(wid, e["id"], e["kind"], e["name"]) for e in w["entities"]],
        ),
        (
            "facts",
            "world_id, fact_id, subj, pred, obj",
            [(wid, f["id"], f["subj"], f["pred"], Jsonb(f["obj"])) for f in w["facts"]],
        ),
        (
            "sources",
            "world_id, source_id, title, reliability, body, fact_ids, false_claims, copy_of",
            [_source_row(wid, s) for s in batch.sources],
        ),
        (
            "questions",
            "world_id, question_id, body, primitive, hops, answerable, answer, missing,"
            " required_facts, required_sources, trap, expected_exit",
            [_question_row(wid, q) for q in batch.questions],
        ),
    ]


def _source_row(wid: str, s: dict) -> tuple:
    return (
        wid,
        s["source_id"],
        s["title"],
        s["reliability"],
        s["text"],
        s["facts"],
        Jsonb(s["false_claims"]),
        s["copy_of"],
    )


def _question_row(wid: str, q: dict) -> tuple:
    return (
        wid,
        q["question_id"],
        q["text"],
        q["primitive"],
        q["hops"],
        q["answerable"],
        q["answer"],
        q.get("missing"),
        q["required_facts"],
        q["required_sources"],
        q["trap"],
        q["expected_exit"],
    )


def _copy(cur: psycopg.Cursor, table: str, columns: str, rows: list[tuple]) -> None:
    # COPY streams all rows in one round trip, unlike row-by-row INSERTs.
    with cur.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
        for row in rows:
            copy.write_row(row)


def load_files(conn: psycopg.Connection, paths: list[Path], prune: bool = False) -> int:
    """Loads every world of `paths` in one transaction, so a bad line leaves the
    database exactly as it was; returns the number of worlds loaded.

    With `prune`, worlds absent from `paths` are deleted (the real-text probe
    excepted), so the database mirrors the files exactly. Off by default:
    loading test.jsonl alone must not wipe the train split.
    """
    seen: set[str] = set()
    with conn.transaction():
        for path in paths:
            for batch in group_worlds(read_records(path)):
                world_id = batch.world["world_id"]
                if world_id in seen:
                    raise ValueError(f"{batch.where}: {world_id} already loaded from another file")
                seen.add(world_id)
                load_world(conn, batch)
        if prune:
            # Pruning against nothing would empty the whole database: an empty
            # or truncated input file is far likelier than a wish to wipe it.
            if not seen:
                raise ValueError("no world read: refusing to prune the database against nothing")
            # The real-text probe loads from its own folder (realtext.load):
            # a reload of the invented worlds must not wipe it.
            conn.execute(
                "DELETE FROM worlds WHERE split <> 'real' AND NOT (world_id = ANY(%s))",
                [list(seen)],
            )
    return len(seen)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m docta_orchestrator.load", description="Load JSONL v2 files into Postgres."
    )
    parser.add_argument("files", nargs="+", type=Path, help="JSONL files from docta-worldgen")
    parser.add_argument(
        "--prune", action="store_true", help="delete worlds absent from FILES (mirror them)"
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("error: DATABASE_URL is not set (see .env.example)", file=sys.stderr)
        return 2
    try:
        with psycopg.connect(url, autocommit=True) as conn:
            loaded = load_files(conn, args.files, prune=args.prune)
    except (OSError, ValueError, psycopg.Error) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"loaded {loaded} worlds", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
