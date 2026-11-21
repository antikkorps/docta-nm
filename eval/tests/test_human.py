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
DUG_KORTHA = {
    "states": [{"state": "plan", "queries": ["q"]}, {"state": "plan", "queries": ["Kortha"]}]
}
RUNS = {
    "none": ({K1: said(None), K2: said("300")}, {K1: NOT_HEARD, K2: DUG_KORTHA}),
    "right": ({K1: said("1432"), K2: said("300")}, {K1: heard("Varnelle"), K2: heard("Kortha")}),
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


def test_reoriented_counts_only_searches_the_model_would_not_have_made_alone():
    s = human_scores(GOLDS, RUNS, HINTS)
    # q2 dug Kortha with no hint too: hearing it changed nothing.
    assert (s["right"]["heard"], s["right"]["reoriented"]) == (1.0, 0.5)
    assert (s["wrong"]["heard"], s["wrong"]["reoriented"]) == (1.0, 0.5)


def test_fooled_counts_heard_runs_whose_bait_differs_from_the_answer():
    hints = {**HINTS, K2: {**HINTS[K2], "wrong_answer": "300"}}  # bait = gold: no bait
    runs = {**RUNS, "wrong": (RUNS["wrong"][0], {K1: heard("Brask"), K2: NOT_HEARD})}
    assert human_scores(GOLDS, runs, hints)["fooled"] == 1.0  # q1 only, and it was fooled
