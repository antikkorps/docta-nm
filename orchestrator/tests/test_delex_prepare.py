"""Preparing a rewritten domain: a third of the asked facts cut from the
texts, every shape a third unanswerable, natural and redacted told apart."""

import json

from docta_orchestrator.delex import prepare


def uni(i, year):
    return {"qid": f"Q{i}", "kind": "university", "label": f"Name{i} University",
            "title": f"Name{i} University", "facts": {"inception": [year]}}  # fmt: skip


def art(i, year):
    return {"qid": f"Q{i}", "sections": [{"path": ["T"], "paragraphs": [
        f"Name{i} University was founded in {year}. It grew fast."]}]}  # fmt: skip


def test_a_domain_gets_a_third_unanswerable_per_shape(tmp_path):
    out = tmp_path / "rewritten"
    out.mkdir()
    ents = [uni(i, 1700 + 10 * i) for i in range(1, 13)]
    arts = [art(i, 1700 + 10 * i) for i in range(1, 13)]
    (out / "entities.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ents))
    (out / "articles.jsonl").write_text("".join(json.dumps(a) + "\n" for a in arts))
    prepare.main([str(out)])
    qs = [json.loads(x) for x in (out / "questions.jsonl").read_text().splitlines()]
    for shape in ("retrieve", "compare"):
        mine = [q for q in qs if q["shape"] == shape]
        no = [q for q in mine if not q["answerable"]]
        assert mine and abs(len(no) / len(mine) - 1 / 3) < 0.1, shape
    assert {q["unanswerable"] for q in qs if not q["answerable"]} == {"redacted"}
    assert (out / "cuts.jsonl").read_text()


def test_balance_holds_per_fact_so_no_wording_predicts_abstention():
    def q(i, fact, ok):
        return {"question_id": f"q{i:03d}", "shape": "retrieve", "fact": fact, "answerable": ok}

    qs = [q(i, "students", i < 3) for i in range(30)] + [q(100 + i, "inception", i % 3 != 0)
                                                         for i in range(30)]  # fmt: skip
    kept = prepare.balanced(qs, 1 / 3, seed=0)
    for fact in ("students", "inception"):
        mine = [x for x in kept if x["fact"] == fact]
        assert abs(sum(not x["answerable"] for x in mine) / len(mine) - 1 / 3) < 0.15, fact


def test_a_cut_reaches_sentences_elsewhere_that_state_the_value():
    ents = [uni(1, 1710), uni(2, 1720)]
    arts = [art(1, 1710), {"qid": "Q2", "sections": [{"path": ["T"], "paragraphs": [
        "Name2 University was founded in 1720. Name1 University dates from 1710."]}]}]  # fmt: skip
    cuts = prepare.redact(ents, arts, share=1.0, seed=1)
    cut = {(c["subject"], c["article"]) for c in cuts}
    assert {("Q1", "Q1"), ("Q1", "Q2")} <= cut
