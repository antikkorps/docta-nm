"""A split with no example stops training with a message saying how to get
data, not with a division by zero hours later (`just gen 42 50` can leave
val or ood empty)."""

from dataclasses import replace

import pytest
from docta_train.fit import fit
from docta_train.fit_read import fit_read
from test_fit import TINY
from test_fit import examples as evaluate_examples
from test_fit_read import examples as read_examples


@pytest.mark.parametrize("empty", ["train", "val"])
def test_evaluate_refuses_an_empty_split(tmp_path, empty):
    train = [] if empty == "train" else evaluate_examples("train", 8)
    val = [] if empty == "val" else evaluate_examples("val", 4)
    with pytest.raises(ValueError, match=f"no {empty} example"):
        fit(train, val, tmp_path, TINY, epochs=1, batch=4, lr=1e-3, seed=0)


@pytest.mark.parametrize("empty", ["train", "val"])
def test_read_refuses_an_empty_split(tmp_path, empty):
    train = [] if empty == "train" else read_examples("train", 8)
    val = [] if empty == "val" else read_examples("val", 4)
    with pytest.raises(ValueError, match=f"no {empty} example"):
        fit_read(train, val, tmp_path, TINY, epochs=1, batch=4, lr=1e-3, seed=0)


def test_the_evaluate_cli_says_data_is_missing_not_balancing(tmp_path, monkeypatch, capsys):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(evaluate_examples("train", 8), tmp_path / "train.jsonl")
    write_examples([], tmp_path / "val.jsonl")
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    assert fit_module.main(["--data", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "no val example" in err and "balancing drops" not in err


def test_read_refuses_a_train_split_with_no_answer_to_point_at(tmp_path):
    # Every answer missing from its excerpts: nothing to learn from, and the
    # epoch's mean loss would divide by zero.
    lost = [replace(e, answer="9999") for e in read_examples("train", 8)]
    with pytest.raises(ValueError, match="no train example"):
        fit_read(lost, read_examples("val", 4), tmp_path, TINY,
                 epochs=1, batch=4, lr=1e-3, seed=0)  # fmt: skip
