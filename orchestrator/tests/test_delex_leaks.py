"""The leak check: what of the real article survives its rewriting, and a
build refused past the thresholds."""

from docta_orchestrator.delex import leaks

ORIGINAL = "The Old Bridge, 120 m long, met Brask in 1901."
RAW = {"qid": "Q1", "links": ["Varn River"],
       "sections": [{"path": ["T"], "paragraphs": [ORIGINAL]}]}  # fmt: skip
ENT = {
    "qid": "Q1",
    "label": "Old Bridge",
    "aliases": [],
    "facts": {"opened": [1901], "length_m": [120]},
}


def article(text):
    return {"qid": "Q1", "sections": [{"path": ["T"], "paragraphs": [text]}]}


def test_surviving_values_names_and_capitalised_words_are_counted():
    found = leaks.of_article(RAW, ENT, article("The Old Bridge, 120 m long, met Kelda in 1801."))
    assert found["values"] == [120]
    assert found["subject"] == "Old Bridge"


def test_a_clean_article_leaks_nothing():
    clean = article("The Kelda span, 180 m long, opened in 1801 in July.")
    found = leaks.of_article(RAW, ENT, clean)
    assert found == {"values": [], "subject": None, "missed": 0}


def test_the_report_refuses_a_corpus_past_its_thresholds():
    clean = {"values": [], "subject": None, "missed": 0}
    dirty = {"values": [120], "subject": None, "missed": 0}
    report = leaks.report([(clean, 2), (dirty, 2)], words=1000)
    assert report["value_rate"] == 0.25 and not report["passed"]
    assert leaks.report([(clean, 2)] * 3, words=1000)["passed"]


def test_a_value_counts_only_where_it_stayed_in_place():
    # Another event's year, shifted, may land on the real founding year:
    # the value is there, but nothing of the original stayed in place.
    moved = article("The Kelda span, 180 m long, met Rosta in 1801; renovated in 1901.")
    found = leaks.of_article(RAW, ENT, moved)
    assert found["values"] == []


def test_a_value_the_rewriter_never_saw_is_a_leak():
    raw = {**RAW, "sections": [{"path": ["T"], "paragraphs": ["Opened in AD1901 by Brask."]}]}
    found = leaks.of_article(raw, ENT, article("Opened in AD1901 by Kelda."))
    assert found["values"] == [1901]
    assert leaks.missed(raw) == 1


def test_small_values_that_read_as_days_are_not_checked():
    ent = {**ENT, "facts": {"spans": [8]}}
    found = leaks.of_article(RAW, ent, article("Opened on 8 August."))
    assert found["values"] == []
