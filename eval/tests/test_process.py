import pytest
from docta_eval.metrics import Gold
from docta_eval.process import process_scores
from eval_sample import GOLD, NAMES
from eval_traces import TRACES, rounds, trace

GOLDS = [Gold.from_record(q, NAMES) for q in GOLD]


def by_key(traces):
    return {(t["world_id"], t["question_id"]): t for t in traces}


def test_scores_the_reflection_not_the_answer():
    assert process_scores(GOLDS, by_key(TRACES)) == pytest.approx(
        {
            "traces": 5,
            "missing": 0,
            "reading_recall": 4 / 5,  # s1 | s2 of s2,s1 | s3,s4
            "reading_precision": 4 / 6,  # read s1 | s2,s9 | s3,s4,s5
            "mean_rounds": 6 / 5,  # q5's plan had no query: not a search round
            "conclusion_rate": 1.0,  # q1, q2, q3 all concluded
            "early_conclusion_rate": 1 / 3,  # q2
            "late_conclusion_rate": 1 / 3,  # q3
            "searched_before_abstaining": 1 / 2,  # q4 yes, q5 no
            "grounded_abstention_rate": 1 / 2,  # q4 read about Brask; q5 read nothing
        }
    )


def test_an_abstention_without_search_is_not_grounded():
    no_search = [t for t in TRACES if t["question_id"] in ("q4", "q5")]
    lazy = [{**t, "states": [t["states"][0], *t["states"][-1:]]} for t in no_search]
    scores = process_scores(GOLDS, by_key(lazy))
    assert scores["grounded_abstention_rate"] == 0.0
    assert scores["searched_before_abstaining"] == 0.0


def test_missing_traces_are_counted_and_answer_metrics_undefined():
    scores = process_scores(GOLDS, by_key(TRACES[3:]))
    assert scores["missing"] == 3
    assert scores["reading_recall"] is None
    assert scores["early_conclusion_rate"] is None


def test_searching_off_subject_is_not_grounded():
    # s7 names nothing the question asks about: a search, but not about Brask.
    off_topic = by_key([trace("q4", rounds(["s7"]), True)])
    scores = process_scores(GOLDS, off_topic)
    assert scores["searched_before_abstaining"] == 1.0
    assert scores["grounded_abstention_rate"] == 0.0


def test_a_missing_trace_leaves_the_grounded_denominator_alone():
    scores = process_scores(GOLDS, by_key(TRACES[:4]))
    assert scores["grounded_abstention_rate"] == 1.0  # q4 of q4 only


def test_subjects_are_the_world_entities_the_question_names():
    assert GOLDS[3].subjects == {"Brask"}
    assert GOLDS[4].subjects == {"Orlan Teste"}
