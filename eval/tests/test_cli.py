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
