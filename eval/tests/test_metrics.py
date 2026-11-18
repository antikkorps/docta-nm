import pytest
from docta_eval.metrics import Gold, Pred, normalize, score
from eval_sample import GOLD, MIXED, pred

GOLDS = [Gold.from_record(q) for q in GOLD]


def preds(records):
    return {(p["world_id"], p["question_id"]): Pred.from_record(p) for p in records}


def test_normalize_ignores_case_spacing_and_final_period():
    assert normalize("  Varnelle  Sud. ") == normalize("varnelle sud") == "varnelle sud"


def test_overall_scores_of_a_mixed_model():
    overall = score(GOLDS, preds(MIXED))["overall"]
    assert overall == pytest.approx(
        {
            "questions": 5,
            "missing": 0,
            "accuracy": 2 / 3,
            "justified_accuracy": 1 / 3,  # q1; q2 is right but cites only s2 of s2, s1
            "citation_precision": 2 / 3,  # s1, s2 right; s9 wrong
            "citation_recall": 2 / 5,  # of s1 | s2 s1 | s3 s4
            "abstention_precision": 1 / 2,  # q4 right, q3 wrong
            "abstention_recall": 1 / 2,  # q4 caught, q5 missed
            "abstention_f1": 1 / 2,
            "false_abstention_rate": 1 / 3,  # q3 of q1, q2, q3
        }
    )


def test_breakdown_by_primitive_and_hops():
    s = score(GOLDS, preds(MIXED))
    retrieve, compare, missing = (
        s["by_primitive"][p] for p in ["retrieve", "compare", "spot_missing"]
    )
    assert retrieve["accuracy"] == 1.0
    assert retrieve["abstention_precision"] is None  # never abstained: undefined
    assert retrieve["abstention_f1"] is None
    assert missing["accuracy"] is None  # nothing answerable
    assert missing["justified_accuracy"] is None
    assert missing["abstention_f1"] == pytest.approx(2 / 3)
    assert s["by_hops"]["1"]["accuracy"] == 0.5
    assert s["by_hops"]["2"]["abstention_recall"] == 0.0


def test_perfect_and_always_abstain_models():
    perfect = [pred("q1", "1432", ["s1"]), pred("q2", "Varnelle", ["s2", "s1"]),
               pred("q3", "Brask", ["s3", "s4"]), pred("q4", abstained=True),
               pred("q5", abstained=True)]  # fmt: skip
    top = score(GOLDS, preds(perfect))["overall"]
    assert [
        top[k]
        for k in [
            "accuracy",
            "justified_accuracy",
            "citation_precision",
            "citation_recall",
            "abstention_f1",
        ]
    ] == [1, 1, 1, 1, 1]
    abstain = score(GOLDS, preds([pred(q["question_id"], abstained=True) for q in GOLD]))["overall"]
    assert abstain["accuracy"] == 0.0
    assert abstain["abstention_recall"] == 1.0
    assert abstain["abstention_precision"] == pytest.approx(2 / 5)
    assert abstain["citation_precision"] is None  # cited nothing


def test_a_missing_prediction_is_wrong_and_not_an_abstention():
    overall = score(GOLDS, preds(MIXED[:4]))["overall"]
    assert overall["missing"] == 1
    assert overall["abstention_recall"] == 1 / 2  # q5 still missed
    without_q1 = score(GOLDS, preds(MIXED[1:]))["overall"]
    assert without_q1["justified_accuracy"] == 0.0  # q1 missing, q2 not justified


def test_abstention_precision_is_undefined_where_no_question_lacks_an_answer():
    # One wrong abstention among answerable questions must not read as "F1 0".
    s = score(GOLDS, preds(MIXED))
    compare = s["by_primitive"]["compare"]
    assert compare["abstention_precision"] is None
    assert compare["abstention_f1"] is None
    assert compare["false_abstention_rate"] == 1.0  # q3, its only question
    assert s["overall"]["false_abstention_rate"] == pytest.approx(1 / 3)


def test_no_gold_question_gives_empty_but_valid_scores():
    assert score([], {})["overall"]["questions"] == 0


def test_a_right_answer_is_justified_only_by_every_source_it_needs():
    # Answering because the evidence establishes it, not because it is known:
    # a right answer without its sources is a lucky guess.
    def justified(cited):
        return score(GOLDS[1:2], preds([pred("q2", "Varnelle", cited)]))["overall"]

    assert justified([])["justified_accuracy"] == 0.0
    assert justified(["s1"])["justified_accuracy"] == 0.0
    assert justified(["s1", "s2", "s9"])["justified_accuracy"] == 1.0  # extra: still justified
    assert justified([])["accuracy"] == 1.0
