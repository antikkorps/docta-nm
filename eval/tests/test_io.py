import json

import pytest
from docta_eval.io import read_gold, read_predictions
from eval_sample import GOLD, MIXED, pred

WORLD = {
    "schema": "v1", "type": "world", "world_id": "w1", "seed": 1, "split": "test",
    "entities": [], "facts": [],
}  # fmt: skip
KEYS = {("w1", q["question_id"]) for q in GOLD}


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_gold_keeps_only_the_questions(tmp_path):
    golds = read_gold(write(tmp_path / "test.jsonl", [WORLD, *GOLD]))
    assert [g.question_id for g in golds] == ["q1", "q2", "q3", "q4", "q5"]


def test_reads_predictions_by_world_and_question(tmp_path):
    preds = read_predictions(write(tmp_path / "m.jsonl", MIXED), KEYS)
    assert preds[("w1", "q4")].abstained
    assert preds[("w1", "q1")].cited_sources == {"s1"}


@pytest.mark.parametrize(
    ("records", "error"),
    [
        ([pred("q1"), pred("q1")], r"m\.jsonl:2: duplicate prediction for w1/q1"),
        ([pred("q9")], r"m\.jsonl:1: w1/q9 is not a gold question"),
        (
            [{k: v for k, v in pred("q1").items() if k != "cited_sources"}],
            r"m\.jsonl:1:.*cited_sources",
        ),
    ],
)
def test_rejects_bad_predictions_with_their_line(tmp_path, records, error):
    with pytest.raises(ValueError, match=error):
        read_predictions(write(tmp_path / "m.jsonl", records), KEYS)


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"answer": 1432}, "answer must be a string or null"),
        ({"cited_sources": "s1"}, "cited_sources must be a list of strings"),
        ({"cited_sources": None}, "cited_sources must be a list of strings"),
        ({"abstained": "false"}, "abstained must be true or false"),
        ({"world_id": 20}, "world_id must be a string"),
        ({"question_id": None}, "question_id must be a string"),
        ({"abstained": True, "answer": "1432"}, "abstained but also answers"),
    ],
)
def test_rejects_ill_typed_predictions(tmp_path, change, error):
    record = {**pred("q1", "1432", ["s1"]), **change}
    with pytest.raises(ValueError, match=rf"m\.jsonl:1: .*{error}"):
        read_predictions(write(tmp_path / "m.jsonl", [record]), KEYS)


def test_rejects_a_gold_file_without_questions(tmp_path):
    with pytest.raises(ValueError, match="no question"):
        read_gold(write(tmp_path / "test.jsonl", [WORLD]))


def test_rejects_an_answerable_gold_question_without_answer(tmp_path):
    broken = {**GOLD[0], "answer": None}
    with pytest.raises(ValueError, match=r"test\.jsonl:1: .*answerable but has no answer"):
        read_gold(write(tmp_path / "test.jsonl", [broken]))


def test_rejects_a_gold_question_listed_twice(tmp_path):
    with pytest.raises(ValueError, match=r"test\.jsonl:2: .*w1/q1 twice"):
        read_gold(write(tmp_path / "test.jsonl", [GOLD[0], GOLD[0]]))


def test_reads_traces_and_rejects_bad_ones(tmp_path):
    from docta_eval.io import read_traces
    from eval_traces import TRACES

    assert len(read_traces(write(tmp_path / "m.jsonl", TRACES), KEYS)) == 5
    no_states = {k: v for k, v in TRACES[0].items() if k != "states"}
    with pytest.raises(ValueError, match=r"m\.jsonl:1: .*states"):
        read_traces(write(tmp_path / "m.jsonl", [no_states]), KEYS)
    stranger = {**TRACES[0], "question_id": "q9"}
    with pytest.raises(ValueError, match=r"m\.jsonl:1: w1/q9 is not a gold question"):
        read_traces(write(tmp_path / "m.jsonl", [stranger]), KEYS)


def frame():
    return {"state": "frame", "question": "q", "unknowns": [], "exit": None}


def concluded(states):
    from eval_traces import TRACES

    return {**TRACES[0], "states": states}


@pytest.mark.parametrize(
    ("states", "error"),
    [
        ([], "must start with a frame"),
        ([{"state": "plan", "round": 1, "queries": []}], "must start with a frame"),
        ([frame(), {"foo": 1}], "event 2 has no known state"),
        ([frame(), {"state": "dream"}], "event 2 has no known state"),
        ([frame(), {"state": "consult", "queries": [], "excerpts": []}], "consult before any plan"),
        (
            [frame(), {"state": "plan", "queries": ["x"]}, {"state": "consult", "excerpts": [{}]}],
            "excerpt without source_id",
        ),
        ([frame(), {"state": "evaluate", "enough": False, "exit": "not_found"}], "disagrees"),
    ],
)
def test_rejects_malformed_traces(tmp_path, states, error):
    from docta_eval.io import read_traces

    with pytest.raises(ValueError, match=rf"m\.jsonl:1: .*{error}"):
        read_traces(write(tmp_path / "m.jsonl", [concluded(states)]), KEYS)


def test_gold_questions_know_the_entities_they_name(tmp_path):
    world = {**WORLD, "entities": [{"id": "e1", "kind": "city", "name": "Brask"}]}
    golds = read_gold(write(tmp_path / "test.jsonl", [world, *GOLD]))
    assert golds[3].subjects == {"Brask"}
