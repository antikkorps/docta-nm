"""The method step by step: one row per step, test/val/ood side by side, and
how far each moved since the last release's snapshot."""

import json

import pytest
from docta_eval.steps import STEPS, against, main, render, step_table


def scores(read_recall=0.8, f1=0.7, chain=0.5):
    """One model's scores.json entry, with only what the steps read."""
    return {
        "overall": {"abstention_f1": f1, "false_abstention_rate": 0.1,
                    "citation_precision": 0.9, "justified_accuracy": 0.6},
        "by_primitive": {"chain": {"justified_accuracy": chain},
                         "retrieve": {"accuracy": 0.75}},
        "traps": {"all": {"accuracy": 0.4, "fooled_rate": 0.3}},
        "process": {"reading_recall": read_recall, "reading_precision": 0.3,
                    "dig_recovery_rate": 0.2, "early_conclusion_rate": 0.05},
    }  # fmt: skip


def test_every_step_of_the_method_has_a_row():
    assert [s for s, _, _ in STEPS] == [
        "search", "judge", "read", "dig", "confront", "conclude", "human"
    ]  # fmt: skip


def test_a_step_reads_its_scores_on_each_split():
    table = step_table({"test": scores(0.8), "ood": scores(0.5)})
    assert table["search"]["reading_recall"] == {"test": 0.8, "ood": 0.5}
    assert table["dig"]["chain_justified_accuracy"] == {"test": 0.5, "ood": 0.5}
    assert table["read"]["retrieve_accuracy"] == {"test": 0.75, "ood": 0.75}


def test_a_split_never_run_is_left_out_not_zero():
    table = step_table({"test": scores(), "val": None})
    assert "val" not in table["judge"]["abstention_f1"]


def test_human_scores_come_from_the_protocol_run():
    table = step_table({"test": scores()}, human={"test": {"recovered": 0.6, "fooled": 0.0}})
    assert table["human"]["recovered"] == {"test": 0.6}
    assert table["human"]["fooled"] == {"test": 0.0}


def test_a_metric_the_model_lacks_is_undefined():
    # rule baselines have no traps section on some splits
    bare = scores()
    del bare["traps"]
    assert step_table({"test": bare})["confront"]["trap_accuracy"] == {"test": None}


def test_the_change_since_a_snapshot_is_per_split():
    now = step_table({"test": scores(f1=0.75), "ood": scores(f1=0.5)})
    then = step_table({"test": scores(f1=0.70)})
    moved = against(now, then)
    assert moved["judge"]["abstention_f1"]["test"] == pytest.approx(0.05)
    assert moved["judge"]["abstention_f1"]["ood"] is None  # nothing to compare


def test_render_shows_each_step_its_splits_and_the_change():
    now = step_table({"test": scores(f1=0.75)})
    text = render(now, against(now, step_table({"test": scores(f1=0.70)})), ["test"])
    line = next(row for row in text.splitlines() if "abstention_f1" in row)
    assert "judge" in line and "0.750" in line and "+0.050" in line


def write_scores(root, split, model, entry):
    (root / split).mkdir(parents=True)
    (root / split / "scores.json").write_text(json.dumps({"gold": "x", "models": {model: entry}}))


def test_cli_saves_a_snapshot_and_compares_with_the_latest(tmp_path, capsys):
    write_scores(tmp_path / "p", "test", "core_read", scores(f1=0.70))
    snaps = tmp_path / "scores"
    args = ["--predictions", str(tmp_path / "p"), "--splits", "test", "--snapshots", str(snaps)]
    assert main([*args, "--save", "0.0.2", "--note", "Evaluate seed 4"]) == 0
    saved = json.loads((snaps / "steps-0.0.2.json").read_text())
    assert saved["version"] == "0.0.2" and "core_read" in saved["models"]
    assert saved["note"] == "Evaluate seed 4"  # which checkpoints made it

    (tmp_path / "p" / "test" / "scores.json").unlink()
    (tmp_path / "p" / "test").rmdir()
    write_scores(tmp_path / "p", "test", "core_read", scores(f1=0.80))
    capsys.readouterr()
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "since 0.0.2" in out and "+0.100" in out


def test_cli_names_a_model_missing_from_the_scores(tmp_path, capsys):
    write_scores(tmp_path / "p", "test", "dig", scores())
    assert main(["--predictions", str(tmp_path / "p"), "--splits", "test",
                 "--snapshots", str(tmp_path / "s")]) == 1  # fmt: skip
    assert "core_read" in capsys.readouterr().err


def test_one_snapshot_per_version_keeps_every_model_saved(tmp_path):
    write_scores(tmp_path / "p", "test", "core_read", scores())
    data = json.loads((tmp_path / "p" / "test" / "scores.json").read_text())
    data["models"]["core_bridge"] = scores(f1=0.5)
    (tmp_path / "p" / "test" / "scores.json").write_text(json.dumps(data))
    args = ["--predictions", str(tmp_path / "p"), "--splits", "test",
            "--snapshots", str(tmp_path / "s"), "--save", "0.0.2"]  # fmt: skip
    assert main(args) == 0 and main([*args, "--model", "core_bridge"]) == 0
    saved = json.loads((tmp_path / "s" / "steps-0.0.2.json").read_text())
    assert set(saved["models"]) == {"core_read", "core_bridge"}


def test_a_split_without_a_human_run_is_blank_not_undefined():
    table = step_table({"test": scores(), "val": scores()}, human={"test": {"recovered": 0.6}})
    assert table["human"]["recovered"] == {"test": 0.6}
    text = render(table, None, ["test", "val"])
    line = next(row for row in text.splitlines() if "recovered" in row)
    assert "—" not in line


def snapshot_args(tmp_path, *extra):
    return ["--predictions", str(tmp_path / "p"), "--splits", "test",
            "--snapshots", str(tmp_path / "s"), *extra]  # fmt: skip


def test_saving_another_model_keeps_the_note(tmp_path):
    write_scores(tmp_path / "p", "test", "core_read", scores())
    assert main(snapshot_args(tmp_path, "--save", "0.0.2", "--note", "seed 4")) == 0
    assert main(snapshot_args(tmp_path, "--save", "0.0.2")) == 0
    assert json.loads((tmp_path / "s" / "steps-0.0.2.json").read_text())["note"] == "seed 4"


def test_the_change_is_against_the_newest_older_version(tmp_path, capsys):
    write_scores(tmp_path / "p", "test", "core_read", scores(f1=0.9))
    s = tmp_path / "s"
    s.mkdir()
    for version, f1 in [("0.0.9", 0.5), ("0.0.10", 0.8), ("0.1.0", 0.1)]:
        table = step_table({"test": scores(f1=f1)})
        doc = {"version": version, "note": "", "models": {"core_read": table}}
        (s / f"steps-{version}.json").write_text(json.dumps(doc))
    assert main(snapshot_args(tmp_path, "--save", "0.0.11")) == 0
    out = capsys.readouterr().out
    assert "since 0.0.10" in out and "+0.100" in out  # never 0.1.0, newer


def test_a_version_not_x_y_z_is_refused(tmp_path, capsys):
    write_scores(tmp_path / "p", "test", "core_read", scores())
    assert main(snapshot_args(tmp_path, "--save", "v0.0.3")) == 1
    assert "X.Y.Z" in capsys.readouterr().err


def test_a_broken_snapshot_is_an_error_not_a_crash(tmp_path, capsys):
    write_scores(tmp_path / "p", "test", "core_read", scores())
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "steps-0.0.1.json").write_text("{not json")
    assert main(snapshot_args(tmp_path)) == 1
    assert "steps-0.0.1.json" in capsys.readouterr().err


def test_real_text_columns_join_by_themselves_once_scored(tmp_path, capsys):
    import json as json_

    from docta_eval.steps import main as steps_main

    for split in ("test", "real_cities"):
        (tmp_path / split).mkdir()
        scores = {"models": {"core_read": {"overall": {"abstention_f1": 0.5}}}}
        (tmp_path / split / "scores.json").write_text(json_.dumps(scores))
    assert steps_main(["--predictions", str(tmp_path), "--snapshots", str(tmp_path / "s")]) == 0
    header = capsys.readouterr().out.splitlines()[1]
    assert header.split()[2:] == ["test", "val", "ood", "real_cities"]
