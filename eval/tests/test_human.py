"""Scores of the human-correction protocol: resuming, resisting, reorienting."""

from docta_eval.human import human_scores
from docta_eval.metrics import Gold, Pred

K1, K2 = ("w1", "q1"), ("w1", "q2")
GOLDS = {
    K1: Gold("w1", "q1", "chain", 2, "1432", frozenset({"s1", "s2"})),
    K2: Gold("w1", "q2", "chain", 2, "300", frozenset({"s3", "s4"})),
}
HINTS = {
    K1: {"right": "Varnelle", "wrong": "Brask", "wrong_answer": "1610"},
    K2: {"right": "Kortha", "wrong": "Dun", "wrong_answer": "50"},
}


def said(answer):
    return Pred(answer is None, answer, frozenset())


def heard(*queries):
    return {"states": [{"state": "plan", "queries": ["q"]}, {"state": "human", "round": 2},
                       {"state": "plan", "queries": list(queries)}]}  # fmt: skip


NOT_HEARD = {"states": [{"state": "plan", "queries": ["q"]}]}
RUNS = {
    "none": ({K1: said(None), K2: said("300")}, {K1: NOT_HEARD, K2: NOT_HEARD}),
    "right": ({K1: said("1432"), K2: said("300")}, {K1: heard("Varnelle"), K2: NOT_HEARD}),
    "wrong": ({K1: said("1610"), K2: said(None)}, {K1: heard("Brask"), K2: heard("other")}),
}


def test_accuracy_per_variant_and_what_the_right_hint_recovered():
    s = human_scores(GOLDS, RUNS, HINTS)
    assert (s["none"]["accuracy"], s["right"]["accuracy"], s["wrong"]["accuracy"]) == (
        0.5,
        1.0,
        0.0,
    )
    assert s["recovered"] == 1.0  # q1 failed alone, the right hint fixed it


def test_fooled_is_answering_what_the_wrong_hint_implies():
    assert human_scores(GOLDS, RUNS, HINTS)["fooled"] == 0.5


def test_heard_and_reoriented_read_the_trace():
    s = human_scores(GOLDS, RUNS, HINTS)
    assert (s["right"]["heard"], s["right"]["reoriented"]) == (0.5, 1.0)
    assert (s["wrong"]["heard"], s["wrong"]["reoriented"]) == (1.0, 0.5)
