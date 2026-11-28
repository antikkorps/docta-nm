"""The leak check: what of the real article survives its rewriting, and a
build refused past the thresholds."""

from docta_orchestrator.delex import leaks

RAW = {"qid": "Q1", "links": ["Varn River"], "sections": []}
ENT = {
    "qid": "Q1",
    "label": "Old Bridge",
    "aliases": [],
    "facts": {"opened": [1901], "length_m": [120]},
}


def article(text):
    return {"qid": "Q1", "sections": [{"path": ["T"], "paragraphs": [text]}]}


def test_surviving_values_names_and_capitalised_words_are_counted():
    found = leaks.of_article(RAW, ENT, article("The Old Bridge, 120 m long, met Quebrada in 1801."),
                             invented={"Kelda"}, common={"old", "bridge"})  # fmt: skip
    assert found["values"] == [120]
    assert found["subject"] == "Old Bridge"
    assert found["capitalised"] == ["Quebrada"]


def test_a_clean_article_leaks_nothing():
    clean = article("The Kelda span, 180 m long, opened in 1801 in July.")
    found = leaks.of_article(RAW, ENT, clean, invented={"Kelda"}, common=set())
    assert found == {"values": [], "subject": None, "capitalised": []}


def test_the_report_refuses_a_corpus_past_its_thresholds():
    clean = {"values": [], "subject": None, "capitalised": []}
    dirty = {"values": [120], "subject": None, "capitalised": []}
    report = leaks.report([(clean, 2), (dirty, 2)], words=1000)
    assert report["value_rate"] == 0.25 and not report["passed"]
    assert leaks.report([(clean, 2)] * 3, words=1000)["passed"]
