"""`python -m docta_eval.human`: the protocol's table from its files."""

import json

from docta_eval.human import main

QUESTION = {"schema": "v1", "type": "question", "world_id": "w1", "question_id": "q1",
            "text": "In what year was the city where Orlan Teste lives founded?",
            "primitive": "chain", "hops": 2, "answerable": True, "answer": "1432",
            "missing": None, "required_facts": ["f1", "f2"], "required_sources": ["s1", "s2"],
            "trap": None, "expected_exit": None}  # fmt: skip


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_the_cli_prints_one_row_per_model(tmp_path, capsys):
    write(tmp_path / "test.jsonl", [QUESTION])
    run = tmp_path / "human"
    (run / "traces").mkdir(parents=True)
    write(run / "hints.jsonl", [{"world_id": "w1", "question_id": "q1", "right": "Varnelle",
                                 "wrong": "Brask", "wrong_answer": "1610"}])  # fmt: skip
    for variant, answer in [("none", None), ("right", "1432"), ("wrong", None)]:
        pred = {"world_id": "w1", "question_id": "q1", "abstained": answer is None,
                "answer": answer, "cited_sources": [], "confidence": None,
                "exit": None}  # fmt: skip
        states = [{"state": "frame"}, {"state": "plan", "queries": ["q"]}]
        if answer:
            states.append({"state": "conclude"})
        write(run / f"dig.{variant}.jsonl", [pred])
        write(run / "traces" / f"dig.{variant}.jsonl",
              [{"world_id": "w1", "question_id": "q1", "model": "dig", "question": "q",
                "states": states, "decision": pred}])  # fmt: skip
    assert main([str(tmp_path / "test.jsonl"), str(run)]) == 0
    out = capsys.readouterr().out
    assert "dig" in out and "recovered" in out
