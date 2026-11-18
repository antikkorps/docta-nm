"""The baseline runner: questions in, predictions and traces out."""

import json

from docta_eval.io import read_predictions
from docta_orchestrator.baseline import predict, run_models
from docta_orchestrator.models import AlwaysAbstainModel, KeywordModel
from orch_sample import QUESTIONS
from test_models import fake_search

YEAR = {**QUESTIONS[0], "text": "In what year was Varnelle founded?"}


def test_a_prediction_has_the_brief_fields():
    prediction, trace = predict(YEAR, KeywordModel(), fake_search)
    assert prediction == {
        "question_id": "q1", "world_id": "w_test01", "abstained": False, "answer": "1432",
        "cited_sources": ["s1"], "confidence": 0.5, "exit": None,
    }  # fmt: skip
    assert trace["decision"] == prediction
    assert trace["question"] == YEAR["text"]
    assert [e["state"] for e in trace["states"]][0] == "frame"


def test_an_abstention_carries_its_exit_and_no_answer():
    prediction, _ = predict(YEAR, AlwaysAbstainModel(), fake_search)
    assert prediction["abstained"] is True
    assert (prediction["answer"], prediction["cited_sources"]) == (None, [])
    assert prediction["exit"] == "not_found"


class Spy(AlwaysAbstainModel):
    seen: list[str] = []

    def frame(self, question):
        Spy.seen.append(question)
        return super().frame(question)


def test_the_model_only_ever_sees_the_question_text():
    predict({**YEAR, "answer": "SECRET"}, Spy(), fake_search)
    assert Spy.seen == [YEAR["text"]]


def test_writes_files_the_evaluator_accepts(tmp_path):
    out, traces = tmp_path / "predictions", tmp_path / "traces"
    questions = [YEAR, QUESTIONS[1]]
    run_models(
        questions, [KeywordModel(), AlwaysAbstainModel()], lambda wid: fake_search, out, traces
    )
    keys = {(q["world_id"], q["question_id"]) for q in questions}
    for name in ["keyword", "always_abstain"]:
        assert len(read_predictions(out / f"{name}.jsonl", keys)) == 2
        lines = (traces / f"{name}.jsonl").read_text().splitlines()
        assert [json.loads(line)["question_id"] for line in lines] == ["q1", "q2"]


def test_main_fails_cleanly_without_a_database(monkeypatch, capsys, tmp_path):
    from docta_orchestrator.baseline import main

    split = tmp_path / "test.jsonl"
    split.write_text("")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main([str(split)]) == 2
    monkeypatch.setenv("DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none?connect_timeout=2")
    assert main([str(split), "--out", str(tmp_path / "p"), "--traces", str(tmp_path / "t")]) == 1
    assert "error:" in capsys.readouterr().err


def test_baseline_runs_the_fake_models_and_the_evaluate_rule():
    from docta_orchestrator.baseline import baseline_models

    names = [m.name for m in baseline_models(seed=0)]
    assert names == ["random", "always_abstain", "keyword", "rule_evaluate"]
