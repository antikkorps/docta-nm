import json

from docta_eval.__main__ import main
from eval_sample import GOLD, MIXED, pred


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def test_prints_one_row_per_model_and_exports_json(tmp_path, capsys):
    gold = write(tmp_path / "test.jsonl", GOLD)
    mixed = write(tmp_path / "mixed.jsonl", MIXED)
    abstain = write(
        tmp_path / "abstain.jsonl", [pred(q["question_id"], abstained=True) for q in GOLD]
    )
    out = tmp_path / "scores.json"
    assert main([str(gold), str(mixed), str(abstain), "--json", str(out)]) == 0
    table = capsys.readouterr().out
    assert "mixed" in table and "abstain" in table
    assert "0.667" in table  # mixed accuracy
    assert "—" in table  # undefined metrics are shown, not hidden as 0
    scores = json.loads(out.read_text())
    assert scores["models"]["mixed"]["overall"]["abstention_f1"] == 0.5
    assert scores["models"]["abstain"]["overall"]["citation_precision"] is None
    assert set(scores["models"]["mixed"]["by_primitive"]) == {
        "retrieve",
        "chain",
        "compare",
        "spot_missing",
    }


def test_a_bad_prediction_file_fails_cleanly(tmp_path, capsys):
    gold = write(tmp_path / "test.jsonl", GOLD)
    bad = write(tmp_path / "bad.jsonl", [pred("q9")])
    assert main([str(gold), str(bad)]) == 1
    assert "bad.jsonl:1" in capsys.readouterr().err


def test_two_files_with_one_model_name_are_refused(tmp_path, capsys):
    gold = write(tmp_path / "test.jsonl", GOLD)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first = write(tmp_path / "a" / "mixed.jsonl", MIXED)
    second = write(tmp_path / "b" / "mixed.jsonl", MIXED)
    assert main([str(gold), str(first), str(second)]) == 1
    assert "model 'mixed' appears twice" in capsys.readouterr().err


def test_an_unwritable_json_path_fails_cleanly(tmp_path, capsys):
    gold = write(tmp_path / "test.jsonl", GOLD)
    mixed = write(tmp_path / "mixed.jsonl", MIXED)
    assert main([str(gold), str(mixed), "--json", str(tmp_path / "no" / "s.json")]) == 1
    assert "error:" in capsys.readouterr().err


def test_traces_add_a_reflection_table(tmp_path, capsys):
    from eval_sample import NAMES
    from eval_traces import TRACES

    world = {
        "schema": "v1", "type": "world", "world_id": "w1", "seed": 1, "split": "test",
        "entities": [{"id": f"e{i}", "kind": "city", "name": n} for i, n in enumerate(NAMES)],
        "facts": [],
    }  # fmt: skip
    gold = write(tmp_path / "test.jsonl", [world, *GOLD])
    # Predictions matching the traces: q4 and q5 abstain, the others answer.
    agreeing = []
    for t in TRACES:
        abstained = t["decision"]["abstained"]
        agreeing.append(pred(t["question_id"], None if abstained else "x", abstained=abstained))
    mixed = write(tmp_path / "mixed.jsonl", agreeing)
    (tmp_path / "traces").mkdir()
    write(tmp_path / "traces" / "mixed.jsonl", TRACES)
    out = tmp_path / "scores.json"
    args = [str(gold), str(mixed), "--traces", str(tmp_path / "traces"), "--json", str(out)]
    assert main(args) == 0
    assert "grounded" in capsys.readouterr().out
    process = json.loads(out.read_text())["models"]["mixed"]["process"]
    assert process["grounded_abstention_rate"] == 0.5


def test_a_missing_trace_file_is_an_error(tmp_path, capsys):
    gold = write(tmp_path / "test.jsonl", GOLD)
    mixed = write(tmp_path / "mixed.jsonl", MIXED)
    (tmp_path / "traces").mkdir()
    assert main([str(gold), str(mixed), "--traces", str(tmp_path / "traces")]) == 1
    assert "mixed.jsonl" in capsys.readouterr().err


def test_traces_must_agree_with_predictions(tmp_path, capsys):
    from eval_traces import TRACES

    gold = write(tmp_path / "test.jsonl", GOLD)
    flipped = [{**p, "abstained": not p["abstained"], "answer": None} for p in MIXED]
    mixed = write(tmp_path / "mixed.jsonl", flipped)
    (tmp_path / "traces").mkdir()
    write(tmp_path / "traces" / "mixed.jsonl", TRACES)
    assert main([str(gold), str(mixed), "--traces", str(tmp_path / "traces")]) == 1
    assert "disagrees with its prediction" in capsys.readouterr().err


def test_trap_questions_get_their_own_table(tmp_path, capsys):
    trapped = [{**GOLD[0], "trap": "contradiction"}, *GOLD[1:]]
    gold = write(tmp_path / "test.jsonl", trapped)
    mixed = write(tmp_path / "mixed.jsonl", MIXED)
    out = tmp_path / "scores.json"
    assert main([str(gold), str(mixed), "--json", str(out)]) == 0
    assert "mixed · contradiction" in capsys.readouterr().out
    traps = json.loads(out.read_text())["models"]["mixed"]["traps"]
    assert traps["contradiction"]["accuracy"] == 1.0  # q1 right
    plain = write(tmp_path / "plain.jsonl", GOLD)
    assert main([str(plain), str(mixed)]) == 0
    assert "trap" not in capsys.readouterr().out  # no trap, no table


def traced(tmp_path, preds, traces):
    gold = write(tmp_path / "test.jsonl", GOLD)
    mixed = write(tmp_path / "mixed.jsonl", preds)
    (tmp_path / "traces").mkdir()
    write(tmp_path / "traces" / "mixed.jsonl", traces)
    return main([str(gold), str(mixed), "--traces", str(tmp_path / "traces")])


def decided(p):
    """A well-formed trace whose decision is exactly prediction `p`."""
    from eval_traces import rounds, trace

    return {**trace(p["question_id"], rounds(["s1"]), p["abstained"]), "decision": p}


def test_traces_and_predictions_cover_the_same_questions(tmp_path, capsys):
    # q5 has a trace but no prediction: the two tables would score different sets.
    assert traced(tmp_path, MIXED[:4], [decided(p) for p in MIXED]) == 1
    assert "w1/q5" in capsys.readouterr().err


def test_a_trace_must_agree_on_the_answer_not_only_on_abstaining(tmp_path, capsys):
    traces = [decided({**p, "answer": "other"} if p["question_id"] == "q1" else p) for p in MIXED]
    assert traced(tmp_path, MIXED, traces) == 1
    assert "w1/q1" in capsys.readouterr().err
