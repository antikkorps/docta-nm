"""Redaction: sentences that state a fact are cut from the probe, so a
controlled share of questions has no answer in what the core may read."""

import pytest
from docta_orchestrator.realtext.probe import balanced
from docta_orchestrator.realtext.questions import build
from docta_orchestrator.realtext.redact import apply, redact, tag
from docta_orchestrator.realtext.stated import Index
from test_realtext_questions import ARTICLES, CITIES, article, city


def test_redaction_cuts_every_sentence_that_states_the_fact():
    texts = [*ARTICLES, article("P9", "Varna grew fast after 1749. Rain fell.")]
    cut = redact(CITIES, texts, share=1.0, seed=0)
    varna = [r for r in cut if r["subject"] == "Q1"]
    assert {r["sentence"] for r in varna} == {"Varna was established in 1749.",
                                             "Varna grew fast after 1749."}  # fmt: skip
    assert not Index(apply(texts, cut)).states("Q1", "Varna", [1749])
    assert "Rain fell." in str(apply(texts, cut))  # only the stating sentences go


def test_the_share_counts_natural_absence_first():
    many = [city(f"Q{i}", f"Town{i}", [1700 + i]) for i in range(10)]
    texts = [article(f"Q{i}", f"Town{i} was founded in {1700 + i}.") for i in range(9)]
    # Q9 is already unanswerable: a share of 0.3 needs two more, not three.
    subjects = {r["subject"] for r in redact(many, texts, share=0.3, seed=0)}
    assert len(subjects) == 2 and "Q9" not in subjects


def test_redaction_is_deterministic_and_written_as_records():
    a = redact(CITIES, ARTICLES, share=0.5, seed=4)
    assert a == redact(CITIES, ARTICLES, share=0.5, seed=4)
    assert all(r["type"] == "redaction" and r["schema"] == "real-v0" for r in a)


def test_questions_say_whether_absence_is_natural_or_made():
    cut = redact(CITIES, ARTICLES, share=1.0, seed=0)
    tagged = tag(build(CITIES, ARTICLES, seed=0), build(CITIES, apply(ARTICLES, cut), seed=0))
    kinds = {tuple(q["subjects"]): q["unanswerable"] for q in tagged if q["hops"] == 1}
    assert kinds[("Q2",)] == "natural"  # never stated, before or after
    assert kinds[("Q1",)] == "redacted"
    assert all(q["answer"] is None for q in tagged if q["unanswerable"])


def test_apply_drops_a_paragraph_left_empty_and_keeps_the_original():
    cut = redact(CITIES, ARTICLES, share=1.0, seed=0)
    before = str(ARTICLES)
    out = apply(ARTICLES, cut)
    assert str(ARTICLES) == before
    assert all(p.strip() for a in out for s in a["sections"] for p in s["paragraphs"])


def test_the_cli_writes_the_cuts_and_the_tagged_questions(tmp_path):
    import json

    from docta_orchestrator.realtext import probe

    for name, rows in (("entities", CITIES), ("articles", ARTICLES)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert probe.main([str(tmp_path), "--seed", "3", "--share", "0.5"]) == 0
    cut = [json.loads(line) for line in (tmp_path / "redactions.jsonl").open()]
    assert cut == redact(CITIES, ARTICLES, share=0.5, seed=3)
    asked = [json.loads(line) for line in (tmp_path / "questions.jsonl").open()]
    texts = apply(ARTICLES, cut)
    every = tag(build(CITIES, ARTICLES, seed=3), build(CITIES, texts, seed=3))
    assert asked == balanced(every, share=0.5, seed=3)


def fake(shape, answerable, i):
    return {"question_id": f"q{i:05d}", "shape": shape, "answerable": answerable}


def test_every_shape_ends_with_the_same_share_unanswerable():
    # Compares lose their answer when either side is cut: far more than a
    # third would let the shape alone predict abstention.
    pool = [fake("compare", i % 3 == 0, i) for i in range(30)]  # 10 answerable
    pool += [fake("retrieve", i % 3 != 0, i) for i in range(30, 60)]  # a third already
    kept = balanced(pool, share=1 / 3, seed=0)
    for shape in ("compare", "retrieve"):
        mine = [q for q in kept if q["shape"] == shape]
        assert sum(not q["answerable"] for q in mine) / len(mine) == pytest.approx(1 / 3, abs=0.04)
    assert len([q for q in kept if q["shape"] == "retrieve"]) == 30  # nothing to drop
    assert kept == balanced(pool, share=1 / 3, seed=0)
    assert [q["question_id"] for q in kept] == sorted(q["question_id"] for q in kept)


@pytest.mark.parametrize("share", ["0", "1"])
def test_the_cli_refuses_a_share_with_nothing_to_balance(tmp_path, share):
    from docta_orchestrator.realtext import probe

    with pytest.raises(SystemExit):
        probe.main([str(tmp_path), "--share", share])


def test_a_subject_whose_name_may_hide_elsewhere_is_never_cut():
    # "C" or a second "Salem" can be named in a sentence no check pins on
    # them ("B evolved into C in 1972"): a cut could leave the answer readable.
    short = city("Q5", "C", [1972])
    twins = [city("Q7", "Salem", [1626]), city("Q8", "Salem", [1842])]
    twins[0]["title"], twins[1]["title"] = "Salem, Massachusetts", "Salem, Oregon"
    texts = [article("Q5", "C was made in 1972."), article("Q7", "Salem was settled in 1626."),
             article("Q8", "Salem grew in 1842.")]  # fmt: skip
    assert redact([short, *twins], texts, share=1.0, seed=0) == []


def test_a_shape_with_one_side_only_is_dropped():
    pool = [fake("chain", False, 0), fake("chain", False, 1), fake("retrieve", True, 2),
            fake("retrieve", False, 3)]  # fmt: skip
    assert {q["shape"] for q in balanced(pool, share=1 / 3, seed=0)} == {"retrieve"}


def test_half_a_question_rounds_up_not_to_even():
    pool = [fake("retrieve", True, 0), fake("retrieve", False, 1), fake("retrieve", False, 2)]
    kept = balanced(pool, share=1 / 3, seed=0)
    assert sorted(q["answerable"] for q in kept) == [False, True]
