"""Scores on trap questions: was the model fooled, and did it take the right exit?"""

import pytest
from docta_eval.metrics import Gold, Pred
from docta_eval.traps import trap_scores

CONTRADICTION = Gold(
    "w1", "q1", "retrieve", 1, "1432", frozenset({"s1"}),
    trap="contradiction", false_answers=frozenset({"1450"}),
)  # fmt: skip
CONSENSUS = Gold(
    "w1", "q2", "retrieve", 1, "Varnelle", frozenset({"s2"}),
    trap="false_consensus", false_answers=frozenset({"Brask"}),
)  # fmt: skip
UNDECIDABLE = Gold(
    "w1", "q3", "spot_missing", 1, None, frozenset(),
    trap="undecidable", expected_exit="unresolved_contradiction",
)  # fmt: skip
PLAIN = Gold("w1", "q4", "retrieve", 1, "300", frozenset({"s4"}))
GOLDS = [CONTRADICTION, CONSENSUS, UNDECIDABLE, PLAIN]


def answer(text, cited=(), exit=None):
    return Pred(text is None, text, frozenset(cited), exit)


def test_scores_each_trap_and_who_was_fooled():
    preds = {
        ("w1", "q1"): answer("1432", ["s1"]),  # right, justified
        ("w1", "q2"): answer("Brask", ["s5"]),  # fooled by the copies
        ("w1", "q3"): answer("1610", ["s6"]),  # settled what cannot be settled
        ("w1", "q4"): answer("300", ["s4"]),
    }
    s = trap_scores(GOLDS, preds)
    assert set(s) == {"contradiction", "false_consensus", "undecidable", "all"}
    assert s["contradiction"] == pytest.approx(
        {"questions": 1, "missing": 0, "accuracy": 1.0, "justified_accuracy": 1.0,
         "fooled_rate": 0.0, "exit_precision": None, "exit_recall": None}
    )  # fmt: skip
    assert s["false_consensus"]["fooled_rate"] == 1.0
    assert s["undecidable"]["fooled_rate"] == 1.0  # any answer settles it
    assert s["undecidable"]["accuracy"] is None  # nothing to get right
    assert s["all"]["fooled_rate"] == pytest.approx(2 / 3)


def test_the_right_exit_on_undecidable_and_a_wrong_one_elsewhere():
    preds = {
        ("w1", "q1"): answer(None, exit="unresolved_contradiction"),  # gave up on a decidable one
        ("w1", "q3"): answer(None, exit="unresolved_contradiction"),
    }
    s = trap_scores(GOLDS, preds)
    assert s["undecidable"]["exit_recall"] == 1.0
    assert s["undecidable"]["fooled_rate"] == 0.0
    assert s["contradiction"]["exit_precision"] == 0.0
    assert s["all"]["exit_precision"] == 0.5
    assert s["false_consensus"]["fooled_rate"] == 0.0  # missing: wrong, but not fooled


def test_an_abstention_with_another_exit_misses_the_undecidable_exit():
    preds = {("w1", "q3"): answer(None, exit="not_found")}
    s = trap_scores(GOLDS, preds)["undecidable"]
    assert (s["exit_recall"], s["fooled_rate"]) == (0.0, 0.0)


def test_no_trap_question_gives_no_trap_scores():
    assert trap_scores([PLAIN], {}) == {}


def test_an_unresolved_exit_on_a_plain_question_counts_against_precision():
    preds = {
        ("w1", "q3"): answer(None, exit="unresolved_contradiction"),
        ("w1", "q4"): answer(None, exit="unresolved_contradiction"),  # nothing to resolve
    }
    assert trap_scores(GOLDS, preds)["all"]["exit_precision"] == 0.5


def test_false_answers_match_like_answers_and_a_right_answer_is_never_fooled():
    sloppy = {("w1", "q1"): answer(" 1450. ")}
    assert trap_scores(GOLDS, sloppy)["contradiction"]["fooled_rate"] == 1.0
    same = Gold(**{**CONTRADICTION.__dict__, "false_answers": frozenset({"1432"})})
    right = {("w1", "q1"): answer("1432", ["s1"])}
    assert trap_scores([same], right)["contradiction"]["fooled_rate"] == 0.0


def test_an_abstention_carrying_an_answer_is_not_fooled():
    pred = Pred(True, "1450", frozenset(), "not_found")
    assert trap_scores(GOLDS, {("w1", "q1"): pred})["contradiction"]["fooled_rate"] == 0.0
