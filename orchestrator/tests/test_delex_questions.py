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


def test_a_name_inside_the_subjects_own_name_is_not_asked():
    rosta = {**UNI, "label": "Rosta University", "title": "Rosta University"}
    arts = [art("Q1", "Rosta University has five faculties, founded in 1801.")]
    facts = {q["fact"] for q in questions.build([rosta], arts, seed=0)}
    assert "located_in" not in facts and "inception" in facts


def test_a_value_inside_a_longer_number_is_not_stated():
    uni = {**UNI, "facts": {"students": [126], "inception": [801]}}
    index = questions.Index([art("Q1", "Kelda University has 1,126 students since 1,801 days.")])
    assert questions.stated(index, uni, "students") is None
    assert questions.stated(index, uni, "inception") is None
