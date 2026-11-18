"""Loading into Postgres. Each test rolls back, so the dev database is untouched."""

import json
import os

import psycopg
import pytest
from docta_orchestrator.jsonl import WorldBatch
from docta_orchestrator.load import load_files, load_world
from orch_sample import QUESTIONS, SOURCES, WORLD

TABLES = ["worlds", "entities", "facts", "sources", "questions"]


@pytest.fixture
def conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set: run `just up`, then the tests through `just test`")
    # force_rollback: any transaction the code under test opens becomes a
    # savepoint inside this one, so nothing ever reaches the dev database.
    with psycopg.connect(url) as connection, connection.transaction(force_rollback=True):
        yield connection


def counts(conn, world_id):
    return {
        t: conn.execute(f"SELECT count(*) FROM {t} WHERE world_id = %s", [world_id]).fetchone()[0]
        for t in TABLES
    }


def test_loads_every_record_of_a_world(conn):
    load_world(conn, WorldBatch(WORLD, SOURCES, QUESTIONS))
    assert counts(conn, "w_test01") == {
        "worlds": 1,
        "entities": 2,
        "facts": 2,
        "sources": 2,
        "questions": 2,
    }


def test_keeps_values_and_types(conn):
    load_world(conn, WorldBatch(WORLD, SOURCES, QUESTIONS))
    objs = conn.execute("SELECT obj FROM facts WHERE world_id = 'w_test01' ORDER BY fact_id")
    assert [row[0] for row in objs] == [1432, "e1"]
    q2 = conn.execute(
        "SELECT body, answerable, answer, missing, required_facts FROM questions"
        " WHERE world_id = 'w_test01' AND question_id = 'q2'"
    ).fetchone()
    assert q2 == (QUESTIONS[1]["text"], False, None, QUESTIONS[1]["missing"], [])


def test_sources_are_searchable_by_invented_name(conn):
    load_world(conn, WorldBatch(WORLD, SOURCES, QUESTIONS))
    hits = conn.execute(
        "SELECT source_id FROM sources WHERE world_id = 'w_test01'"
        " AND tsv @@ plainto_tsquery('simple', 'Varnelle') ORDER BY source_id"
    ).fetchall()
    assert hits == [("s1",), ("s2",)]


def test_reloading_a_world_replaces_it(conn):
    load_world(conn, WorldBatch(WORLD, SOURCES, QUESTIONS))
    renamed = [{**SOURCES[0], "title": "Annals of Varnelle"}]
    load_world(conn, WorldBatch(WORLD, renamed, []))
    assert counts(conn, "w_test01")["sources"] == 1
    assert counts(conn, "w_test01")["questions"] == 0
    title = conn.execute("SELECT title FROM sources WHERE world_id = 'w_test01'").fetchone()[0]
    assert title == "Annals of Varnelle"


def test_a_bad_line_leaves_the_database_unchanged(conn, tmp_path):
    good = json.dumps(WORLD) + "\n" + "".join(json.dumps(r) + "\n" for r in SOURCES)
    second = json.dumps({**WORLD, "world_id": "w_test02"}) + "\n{broken\n"
    path = tmp_path / "data.jsonl"
    path.write_text(good + second)
    with pytest.raises(ValueError):
        load_files(conn, [path])
    assert counts(conn, "w_test01")["worlds"] == 0


def jsonl(path, *records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_prune_removes_worlds_absent_from_the_input(conn, tmp_path):
    other = {**WORLD, "world_id": "w_test02"}
    load_files(conn, [jsonl(tmp_path / "both.jsonl", WORLD, other)])
    load_files(conn, [jsonl(tmp_path / "one.jsonl", WORLD)])
    assert counts(conn, "w_test02")["worlds"] == 1, "without --prune, other worlds stay"
    load_files(conn, [jsonl(tmp_path / "one.jsonl", WORLD)], prune=True)
    assert counts(conn, "w_test02")["worlds"] == 0
    assert counts(conn, "w_test01")["worlds"] == 1


def test_a_world_in_two_files_is_an_error(conn, tmp_path):
    a = jsonl(tmp_path / "a.jsonl", WORLD)
    b = jsonl(tmp_path / "b.jsonl", WORLD)
    with pytest.raises(ValueError, match=r"b\.jsonl:1: w_test01 already loaded"):
        load_files(conn, [a, b])
