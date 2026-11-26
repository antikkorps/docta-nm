"""The only window a model has on a world: full-text search over its sources."""

import dataclasses

import pytest
from docta_orchestrator.jsonl import WorldBatch
from docta_orchestrator.load import load_world
from docta_orchestrator.search import Hit, search, to_tsquery
from orch_sample import QUESTIONS, SOURCES, WORLD


def loaded(conn):
    load_world(conn, WorldBatch(WORLD, SOURCES, QUESTIONS))
    return conn


def ids(hits):
    return [h.source_id for h in hits]


def test_hits_expose_text_but_never_ground_truth():
    assert [f.name for f in dataclasses.fields(Hit)] == [
        "source_id",
        "title",
        "body",
        "reliability",
    ]


def test_query_words_are_ored_so_partial_matches_count():
    assert to_tsquery("Where does Orlan Teste live?") == "where | does | orlan | teste | live"
    assert to_tsquery("?! ...") is None


def test_ranks_the_best_matching_source_first(conn):
    assert ids(search(loaded(conn), "w_test01", "Orlan Teste")) == ["s2"]
    assert ids(search(conn, "w_test01", "When was Varnelle founded?")) == ["s1", "s2"]


def test_returns_the_text_a_model_reads(conn):
    hit = search(loaded(conn), "w_test01", "Orlan")[0]
    assert hit == Hit("s2", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.7)


def test_stays_inside_the_asked_world(conn):
    load_world(loaded(conn), WorldBatch({**WORLD, "world_id": "w_test02"}, SOURCES, []))
    assert {h.source_id for h in search(conn, "w_test01", "Varnelle")} == {"s1", "s2"}
    assert search(conn, "w_test03", "Varnelle") == []


def test_respects_k_and_empty_queries(conn):
    assert len(search(loaded(conn), "w_test01", "Varnelle", k=1)) == 1
    assert search(conn, "w_test01", "?!") == []


@pytest.mark.parametrize("k", [0, -1])
def test_a_k_below_one_is_refused_before_any_query(k):
    # LIMIT 0 would read as "nothing found" and a negative one fails in SQL.
    with pytest.raises(ValueError, match="k must be at least 1"):
        search(None, "w_test01", "Varnelle", k=k)


def test_odd_queries_reach_the_database_safely(conn):
    loaded(conn)
    assert ids(search(conn, "w_test01", "Varnelle's founding")) == ["s1", "s2"]
    assert ids(search(conn, "w_test01", "1432")) == ["s1"]
    assert search(conn, "w_test01", "x" * 3000) == []  # too long to index: no match, no error


def test_search_is_never_prepared(conn):
    # A prepared statement gets a generic plan that ignores world_id and scans
    # the whole full-text index: 20x slower once there are thousands of worlds.
    loaded(conn)
    for _ in range(10):
        search(conn, "w_test01", "Varnelle")
    assert conn.execute("SELECT count(*) FROM pg_prepared_statements").fetchone()[0] == 0
