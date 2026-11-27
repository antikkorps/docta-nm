"""The sample Franck checks by hand before any real-text score is believed:
so many questions of each kind, each with what shows its label right or
wrong."""

import json

from docta_orchestrator.realtext import probe
from docta_orchestrator.realtext.sample import KINDS, sample
from docta_orchestrator.realtext.stated import Index
from test_realtext_questions import ARTICLES, CITIES, article


def prepared(tmp_path, share="0.5"):
    rows = [{**r, "domain": "cities"} for r in CITIES]
    for name, data in (("entities", rows), ("articles", ARTICLES)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    assert probe.main([str(tmp_path), "--share", share]) == 0
    return tmp_path


def test_the_sample_draws_up_to_so_many_of_each_kind_the_same_way_each_time(tmp_path):
    where = prepared(tmp_path)
    items = sample([where], per_kind=1, seed=3)
    assert items == sample([where], per_kind=1, seed=3)
    kinds = [i["kind"] for i in items]
    assert set(kinds) <= set(KINDS) and all(kinds.count(k) <= 1 for k in KINDS)
    assert all(i["verdict"] is None and i["domain"] == "cities" for i in items)


def test_an_answerable_question_shows_the_sentence_that_states_it(tmp_path):
    items = sample([prepared(tmp_path)], per_kind=50, seed=0)
    varna = next(i for i in items if i["kind"] == "answerable" and i["subjects"] == ["Q1"])
    assert ["Q1", "Varna was established in 1749."] in varna["evidence"]["states"]


def test_a_redacted_question_shows_the_cut_and_nothing_left_stating_it(tmp_path):
    where = prepared(tmp_path, share="0.67")
    cut = next(i for i in sample([where], per_kind=50, seed=0) if i["kind"] == "redacted")
    assert cut["evidence"]["cut"] and cut["evidence"]["states"] == []


def test_a_natural_absence_shows_the_years_its_article_does_hold(tmp_path):
    where = prepared(tmp_path)
    brask = next(i for i in sample([where], per_kind=50, seed=0)
                 if i["kind"] == "natural" and i["subjects"] == ["Q2"])  # fmt: skip
    assert brask["evidence"]["states"] == [] and brask["values"] == ["1802"]
    assert brask["evidence"]["years_nearby"] == []  # Brask's article holds no year


def test_a_chain_shows_the_sentence_that_links_its_two_entities():
    index = Index([article("P1", "Ada Kel was born in 1950 in Varna.")])
    assert index.linking(("P1", "Ada Kel"), ("Q1", "Varna"), "born_in") == [
        ("P1", "Ada Kel was born in 1950 in Varna.")
    ]
    assert index.links(("P1", "Ada Kel"), ("Q1", "Varna"), "born_in")


def test_the_cli_writes_the_sample(tmp_path):
    from docta_orchestrator.realtext import sample as module

    (tmp_path / "cities").mkdir()
    where = prepared(tmp_path / "cities")
    out = tmp_path / "sample.jsonl"
    assert module.main([str(where), "--per-kind", "2", "--seed", "1", "--out", str(out)]) == 0
    assert [json.loads(line) for line in out.open()] == sample([where], per_kind=2, seed=1)


def chain_domain(tmp_path, texts):
    rows = [{**r, "domain": "cities"} for r in CITIES]
    for name, data in (("entities", rows), ("articles", texts)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    return rows


def test_cuts_shown_are_those_that_took_the_question_s_evidence(tmp_path):
    # The chain asks Varna's year through Ada Kel's birthplace: cutting Ada
    # Kel's own birth year takes nothing from it; cutting the link does.
    from docta_orchestrator.realtext.questions import build
    from docta_orchestrator.realtext.redact import apply

    texts = [*ARTICLES[:3], article("P1", "Ada Kel was born in 1950.", "She was born in Varna.")]
    rows = chain_domain(tmp_path, texts)
    link = ["P1", "She was born in Varna.", "link"]
    for sentence, shown in (("Ada Kel was born in 1950.", []), ("She was born in Varna.", [link])):
        cuts = [{"subject": "P1", "fact": "birth_year", "article": "P1", "sentence": sentence}]
        q = next(q for q in build(rows, apply(texts, cuts)) if q["shape"] == "chain")
        (tmp_path / "questions.jsonl").write_text(json.dumps(q) + "\n")
        (tmp_path / "redactions.jsonl").write_text(json.dumps(cuts[0]) + "\n")
        [it] = [i for i in sample([tmp_path], per_kind=5) if i["shape"] == "chain"]
        assert it["evidence"]["cut"] == shown


def test_a_compare_shows_both_years_and_every_article_quoted_links_out(tmp_path):
    from docta_orchestrator.realtext.questions import build

    texts = [*ARTICLES, article("P9", "Brask grew fast after 1802.")]
    rows = chain_domain(tmp_path, texts)
    q = next(q for q in build(rows, texts) if q["shape"] == "compare" and q["answerable"])
    (tmp_path / "questions.jsonl").write_text(json.dumps(q) + "\n")
    (tmp_path / "redactions.jsonl").write_text("")
    [it] = sample([tmp_path], per_kind=5)
    quoted = {a for a, *_ in it["evidence"]["states"]}
    assert {s for s in q["subjects"]} <= quoted | set(q["subjects"])
    assert quoted <= set(it["evidence"]["urls"])


def test_an_absence_shows_every_number_its_article_holds(tmp_path):
    many = " ".join(f"In {1900 + i} it grew." for i in range(10))  # never 1802
    texts = [*ARTICLES[:1], article("Q2", many), *ARTICLES[2:]]
    chain_domain(tmp_path, texts)
    assert probe.main([str(tmp_path), "--share", "0.5"]) == 0
    brask = next(i for i in sample([tmp_path], per_kind=50) if i["subjects"] == ["Q2"])
    assert len(brask["evidence"]["years_nearby"]) == 10


def test_the_order_domains_are_given_in_does_not_change_the_draw(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    prepared(a), prepared(b)
    assert sample([a, b], per_kind=2, seed=5) == sample([b, a], per_kind=2, seed=5)
