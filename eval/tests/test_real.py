"""Scoring the real-text probe: the same table as the worlds, broken down by
shape and by kind of absence, with "supported" citations in place of the
worlds' fact ids."""

import json

import pytest
from docta_eval.__main__ import evaluate, main
from docta_eval.io import read_gold
from docta_eval.metrics import Gold, Pred, score
from docta_eval.real import READING, load, supported
from docta_orchestrator.realtext.questions import build
from eval_sample import GOLD
from real_sample import ARTICLES, ENTITIES, article, asked, city, domain


def by_body(where, word):
    return next(q for q in asked(where) if word in q["body"] and q["hops"] == 1
                and " or " not in q["body"])  # fmt: skip


def test_a_domain_folder_reads_as_gold_in_its_own_world(tmp_path):
    where = domain(tmp_path)
    golds = {g.question_id: g for g in read_gold(where)}
    assert {g.world_id for g in golds.values()} == {"real_cities"}
    for q in asked(where):
        g = golds[q["question_id"]]
        assert (g.primitive, g.hops, g.shape) == (q["primitive"], q["hops"], q["shape"])
        assert g.answer == (q["answer"] if q["answerable"] else None)
        assert g.absence == (q["unanswerable"] or "answerable")


def test_scores_break_down_by_shape_and_absence_and_blank_fact_citations(tmp_path):
    golds = read_gold(domain(tmp_path))
    scores = score(golds, {})
    assert set(scores["by_shape"]) == {g.shape for g in golds}
    assert set(scores["by_absence"]) == {g.absence for g in golds}
    for field in ("justified_accuracy", "citation_precision", "citation_recall"):
        assert scores["overall"][field] is None  # a paragraph has no fact ids


def test_worlds_keep_their_scores_without_the_real_breakdowns():
    golds = [Gold.from_record(q) for q in GOLD]
    assert "by_shape" not in score(golds, {}) and "by_absence" not in score(golds, {})


def answer(q, value, cited):
    return {(q.world_id, q.question_id): Pred(False, value, frozenset(cited))}


def test_a_right_answer_is_supported_by_a_cited_paragraph_that_states_it(tmp_path):
    where = domain(tmp_path)
    q = by_body(where, "Varna")
    gold = next(g for g in read_gold(where) if g.question_id == q["question_id"])
    own = supported(load(where), [gold], answer(gold, "1749", ["Q1:0"]))
    assert own == {"right": 1, "supported": 1, "supported_rate": 1.0}
    elsewhere = supported(load(where), [gold], answer(gold, "1749", ["Q4:0"]))
    assert elsewhere["supported"] == 0  # Kesh's paragraph says nothing of Varna
    wrong = supported(load(where), [gold], answer(gold, "1650", ["Q4:0"]))
    assert wrong == {"right": 0, "supported": 0, "supported_rate": None}


def test_a_comparison_needs_both_years_backed(tmp_path):
    # Balancing drops a shape with one side only, so the sample is set by hand.
    where = domain(tmp_path)
    q = next(q for q in build(ENTITIES, ARTICLES) if q["shape"] == "compare" and q["answerable"])
    (where / "questions.jsonl").write_text(json.dumps(q) + "\n")
    (where / "redactions.jsonl").write_text("")
    [g] = read_gold(where)
    one, two = (f"{s}:0" for s in q["subjects"])
    assert supported(load(where), [g], answer(g, g.answer, [one]))["supported"] == 0
    assert supported(load(where), [g], answer(g, g.answer, [one, two]))["supported"] == 1


def test_the_cli_scores_a_domain_and_prints_supported(tmp_path, capsys):
    (tmp_path / "cities").mkdir()
    where = domain(tmp_path / "cities")
    preds = tmp_path / "abstain.jsonl"
    preds.write_text("".join(json.dumps({"question_id": q["question_id"],
        "world_id": "real_cities", "abstained": True, "answer": None,
        "cited_sources": []}) + "\n" for q in asked(where)))  # fmt: skip
    out = tmp_path / "scores.json"
    assert main([str(where), str(preds), "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert any(line.startswith("supported ") for line in printed.splitlines())
    models = json.loads(out.read_text())["models"]
    assert models["abstain"]["supported"]["right"] == 0
    assert models["abstain"]["overall"]["abstention_recall"] == 1.0


def hand_set(tmp_path, entities, articles, pick):
    """A domain whose single question is chosen by hand: balancing would drop
    a shape with one side only."""
    for name, rows in (("entities", entities), ("articles", articles)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    q = next(q for q in build(entities, articles) if pick(q))
    (tmp_path / "questions.jsonl").write_text(json.dumps(q) + "\n")
    (tmp_path / "redactions.jsonl").write_text("")
    [g] = read_gold(tmp_path)
    return q, g


def test_a_chain_is_supported_by_the_year_of_the_city_it_leads_to(tmp_path):
    ada = {"domain": "cities", "kind": "person", "qid": "P1", "label": "Ada Kel",
           "title": "Ada Kel", "facts": {"birth_year": [1950]}, "born_in": ["Q1"]}  # fmt: skip
    texts = [*ARTICLES, article("P1", "Ada Kel was born in Varna.")]
    q, g = hand_set(tmp_path, [*ENTITIES, ada], texts, lambda q: q["shape"] == "chain")
    assert supported(load(tmp_path), [g], answer(g, "1749", ["P1:0"]))["supported"] == 0
    assert supported(load(tmp_path), [g], answer(g, "1749", ["Q1:0"]))["supported"] == 1


def test_a_short_name_is_backed_by_its_own_article_only(tmp_path):
    short = city("Q5", "C", [1972])
    texts = [*ARTICLES, article("Q5", "C was made in 1972."), article("Q6", "In 1972 C and B met.")]
    q, g = hand_set(tmp_path, [*ENTITIES, short], texts, lambda q: q["subjects"] == ["Q5"])
    assert supported(load(tmp_path), [g], answer(g, "1972", ["Q6:0"]))["supported"] == 0
    assert supported(load(tmp_path), [g], answer(g, "1972", ["Q5:0"]))["supported"] == 1


def test_a_city_sharing_its_name_is_not_backed_by_its_twin(tmp_path):
    twins = [city("Q7", "Salem", [1626]), city("Q8", "Salem", [1842])]
    twins[0]["title"], twins[1]["title"] = "Salem, Massachusetts", "Salem, Oregon"
    texts = [
        article("Q7", "Salem was settled in 1626."),
        article("Q8", "Salem, in 1626, was empty."),
    ]
    q, g = hand_set(tmp_path, twins, texts, lambda q: q["subjects"] == ["Q7"])
    assert supported(load(tmp_path), [g], answer(g, "1626", ["Q8:0"]))["supported"] == 0


def test_every_value_the_texts_state_is_a_right_answer(tmp_path):
    # The key keeps disagreements on purpose: answering the other stated
    # value is reading, not a mistake.
    tolm = city("Q3", "Tolm", [1749, 1750])
    texts = [
        article("Q3", "Tolm was founded in 1749."),
        article("Q9", "Tolm got its charter in 1750."),
    ]
    q, g = hand_set(tmp_path, [tolm], texts, lambda q: q["subjects"] == ["Q3"])
    scores = score([g], answer(g, "1750", ["Q9:0"]))
    assert scores["overall"]["accuracy"] == 1.0
    assert supported(load(tmp_path), [g], answer(g, "1750", ["Q9:0"]))["supported"] == 1


def test_a_domain_without_entities_is_refused_by_name(tmp_path):
    for name in ("entities", "articles", "redactions", "questions"):
        (tmp_path / f"{name}.jsonl").write_text("")
    with pytest.raises(ValueError, match="no entit"):
        read_gold(tmp_path)


def test_reading_measures_that_need_fact_ids_are_blank_on_real_text(tmp_path):
    # Two rounds of reading, then a conclusion: on the worlds this scores
    # reading precision and timing against required_sources, which real
    # questions do not have.
    (tmp_path / "cities").mkdir()
    where = domain(tmp_path / "cities")
    ids = [q["question_id"] for q in asked(where)]
    pred = {"world_id": "real_cities", "abstained": False, "answer": "0", "cited_sources": ["Q1:0"]}
    rows = [{"question_id": i, **pred} for i in ids]
    (tmp_path / "reader.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    read = {"state": "consult", "excerpts": [{"source_id": "Q1:0"}]}
    plan = {"state": "plan", "queries": ["Varna"]}
    states = [{"state": "frame"}, plan, read, plan, read, {"state": "conclude"}]
    lines = [{"world_id": "real_cities", "question_id": i, "states": states,
              "decision": {"abstained": False}} for i in ids]  # fmt: skip
    (tmp_path / "traces").mkdir()
    (tmp_path / "traces" / "reader.jsonl").write_text("".join(json.dumps(t) + "\n" for t in lines))
    process = evaluate(where, [tmp_path / "reader.jsonl"], tmp_path / "traces")["reader"]["process"]
    for field in READING:
        assert process[field] is None
    assert process["mean_rounds"] == 2.0  # what needs no fact id stays
