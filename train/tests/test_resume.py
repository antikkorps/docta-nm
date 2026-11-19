"""Resuming a run: more epochs on our own weights, as if never interrupted."""

import json

import pytest
from docta_train.fit import fit
from test_fit import TINY, examples


def run(out, epochs, resume=False, n_train=32):
    return fit(
        examples("train", n_train), examples("val", 8), out, TINY,
        epochs=epochs, batch=8, lr=3e-3, seed=1, resume=resume,
    )  # fmt: skip


def logged(out):
    lines = (out / "log.jsonl").read_text().splitlines()
    return [{k: v for k, v in json.loads(line).items() if k != "minutes"} for line in lines]


def test_two_epochs_then_one_more_equals_three_at_once(tmp_path):
    at_once = run(tmp_path / "a", 3)
    run(tmp_path / "b", 2)
    resumed = run(tmp_path / "b", 1, resume=True)
    assert resumed["best"] == at_once["best"]
    assert logged(tmp_path / "b") == logged(tmp_path / "a")
    assert [line["epoch"] for line in logged(tmp_path / "b")] == [1, 2, 3]


def test_resume_refuses_other_data(tmp_path):
    run(tmp_path, 1)
    with pytest.raises(ValueError, match="without --resume"):
        run(tmp_path, 1, resume=True, n_train=24)


def test_resume_needs_a_run_to_continue(tmp_path):
    with pytest.raises(ValueError, match="nothing to resume"):
        run(tmp_path, 1, resume=True)


def test_cli_passes_resume_on(tmp_path, monkeypatch):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(examples("train", 4), tmp_path / "train.jsonl")
    write_examples(examples("val", 2), tmp_path / "val.jsonl")
    seen = {}
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(fit_module, "fit", lambda *a, **kw: seen.update(kw) or {})
    fit_module.main(["--data", str(tmp_path), "--no-balance", "--resume"])
    assert seen["resume"] is True
