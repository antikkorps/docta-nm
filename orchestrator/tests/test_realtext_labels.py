"""Label fixes from the hand check of 150 questions: sentences cut at an
initial, a link that names a person by surname, and comparisons whose order
a reader can tell without both exact years."""

from docta_orchestrator.realtext.questions import named, order_readable
from docta_orchestrator.realtext.stated import Index, sentences
from test_realtext_questions import article, city


def test_an_initial_or_an_abbreviation_does_not_end_a_sentence():
    one = "It was created by John G. Kemeny and Thomas E. Kurtz in 1964."
    two = "Named after a retired U.S. Army officer."
    assert sentences(f"{one} {two}") == [one, two]
    assert sentences("Brask grew. It lies on a hill.") == ["Brask grew.", "It lies on a hill."]


def test_a_link_may_name_a_person_by_surname():
    index = Index([article("L1", "True BASIC was developed by Kemeny and Kurtz.")])
    assert index.links(("L1", "True BASIC"), ("P1", "John George Kemeny"), "designed_by")
    assert not index.links(("L1", "True BASIC"), ("P2", "Kemeny"), "born_in")  # no link word


def compare(first, later):
    return {"shape": "compare", "answerable": False, "values": [first["label"]],
            "subjects": [first["qid"], later["qid"]]}  # fmt: skip


def test_an_order_readable_from_the_two_articles_dates_is_flagged():
    # Newsqueak "early 1980s" against an AMOS article dated from 1990 on.
    news, amos = city("L1", "Newsqueak", [1980]), city("L2", "AMOS", [1990])
    texts = [article("L1", "It was developed in the early 1980s."),
             article("L2", "AMOS came out on the Amiga. Version 2 followed in 1994.")]  # fmt: skip
    entities = {e["qid"]: e for e in named([news, amos])}
    assert order_readable(compare(news, amos), entities, Index(texts))


def test_an_order_with_no_date_on_one_side_is_not_readable():
    a, b = city("Q1", "Varna", [1749]), city("Q2", "Brask", [1802])
    texts = [article("Q1", "Varna lies on a river."), article("Q2", "Brask grew in 1820.")]
    entities = {e["qid"]: e for e in named([a, b])}
    assert not order_readable(compare(a, b), entities, Index(texts))


def test_the_probe_keeps_no_comparison_whose_order_reads_off_the_dates():
    from docta_orchestrator.realtext.probe import fair
    from docta_orchestrator.realtext.questions import build

    cities = [city("Q1", "Varna", [1749]), city("Q2", "Brask", [1802])]
    texts = [article("Q1", "Varna was founded in 1749."), article("Q2", "Brask grew in 1820.")]
    asked = build(cities, texts)
    kept = fair(asked, cities, texts)
    assert any(q["shape"] == "compare" for q in asked)
    assert not any(q["shape"] == "compare" for q in kept)
    assert [q for q in kept] == [q for q in asked if q["shape"] != "compare"]
