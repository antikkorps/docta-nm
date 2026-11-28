"""Questions on the delexicalised corpus: a year, a quantity or a name per
fact, answerable when the subject's article states the rewritten value."""

from docta_orchestrator.delex import questions

ROSTA = {"qid": "Q9", "label": "Rosta"}
UNI = {"qid": "Q1", "kind": "university", "label": "Kelda University",
       "title": "Kelda University",
       "facts": {"inception": [1801], "students": [12500], "located_in": [ROSTA]}}  # fmt: skip
UNI2 = {"qid": "Q2", "kind": "university", "label": "Thive College", "title": "Thive College",
        "facts": {"inception": [1901], "students": [], "located_in": []}}  # fmt: skip


def art(qid, *paragraphs):
    return {"qid": qid, "sections": [{"path": ["T"], "paragraphs": list(paragraphs)}]}


ARTS = [art("Q1", "Kelda University was founded in 1801 in Rosta. It has 12,500 students."),
        art("Q2", "Thive College is small.")]  # fmt: skip


def test_each_fact_kind_is_stated_in_its_own_form():
    index = questions.Index(ARTS)
    assert questions.stated(index, UNI, "inception") == "1801"
    assert questions.stated(index, UNI, "students") == "12,500"
    assert questions.stated(index, UNI, "located_in") == "Rosta"
    assert questions.stated(index, UNI2, "inception") is None


def test_questions_use_the_worlds_wordings_where_a_relation_has_one():
    qs = questions.build([UNI, UNI2], ARTS, seed=0)
    by = {(q["subjects"][0], q["fact"]): q for q in qs if q["shape"] == "retrieve"}
    assert (
        by[("Q1", "inception")]["wording"] == "seen" and by[("Q1", "inception")]["answer"] == "1801"
    )
    assert by[("Q1", "students")]["wording"] == "new"
    assert by[("Q1", "located_in")]["answer"] == "Rosta"
    assert not by[("Q2", "inception")]["answerable"]
    assert ("Q2", "students") not in by  # no value in the key: nothing to ask


def test_a_comparison_needs_both_values_apart_and_stated():
    qs = [q for q in questions.build([UNI, UNI2], ARTS, seed=0) if q["shape"] == "compare"]
    assert len(qs) == 1 and qs[0]["values"] == ["Kelda University"]
    assert not qs[0]["answerable"]  # Thive's founding year is in no text
