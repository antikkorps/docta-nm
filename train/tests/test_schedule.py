"""Learning-rate warmup and dropout: a steadier start, less memorising."""

import torch
from docta_train.fit import fit
from docta_train.model import CoreEvaluate
from docta_train.training import Warmup, run_epoch
from test_fit import TINY, examples
from test_model import VOCAB, toy_batches


def test_warmup_climbs_to_the_learning_rate_then_holds():
    warmup = Warmup(lr=1.0, steps=4)
    assert [warmup.next() for _ in range(6)] == [0.25, 0.5, 0.75, 1.0, 1.0, 1.0]


def test_no_warmup_is_the_learning_rate_from_the_first_step():
    assert Warmup(lr=0.5, steps=0).next() == 0.5


def test_an_epoch_trains_at_the_warmup_rate():
    model = CoreEvaluate(TINY, VOCAB)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1.0)
    run_epoch(model, optimiser, toy_batches() * 2, Warmup(lr=1e-2, steps=4))
    assert optimiser.param_groups[0]["lr"] == 0.5e-2


def run(out, epochs, resume=False, **kw):
    return fit(examples("train", 32), examples("val", 8), out, TINY, epochs=epochs, batch=8,
               lr=3e-3, seed=1, resume=resume, warmup=10, **kw)  # fmt: skip


def test_a_run_stopped_during_warmup_resumes_where_it_was(tmp_path):
    at_once = run(tmp_path / "a", 3)  # 4 steps an epoch: warmup ends in the third
    run(tmp_path / "b", 2)
    assert run(tmp_path / "b", 1, resume=True)["best"] == at_once["best"]


def test_cli_sets_dropout_and_warmup(tmp_path, monkeypatch):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(examples("train", 4), tmp_path / "train.jsonl")
    write_examples(examples("val", 2), tmp_path / "val.jsonl")
    seen = {}
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(
        fit_module,
        "fit",
        lambda tr, va, out, config, *a, **kw: seen.update(kw, config=config) or {},
    )
    fit_module.main(["--data", str(tmp_path), "--no-balance", "--dropout", "0.2", "--warmup", "7"])
    assert seen["config"].dropout == 0.2 and seen["warmup"] == 7


def test_resume_refuses_a_run_saved_before_warmup(tmp_path):
    import pytest

    run(tmp_path, 1)
    saved = torch.load(tmp_path / "last.pt", weights_only=True)
    del saved["warmup"]
    torch.save(saved, tmp_path / "last.pt")
    with pytest.raises(ValueError, match="without --resume"):
        run(tmp_path, 1, resume=True)


def test_resume_keeps_its_own_warmup_whatever_is_asked(tmp_path):
    at_once = run(tmp_path / "a", 3)
    run(tmp_path / "b", 2)
    resumed = fit(examples("train", 32), examples("val", 8), tmp_path / "b", TINY, epochs=1,
                  batch=8, lr=3e-3, seed=1, resume=True, warmup=999)  # fmt: skip
    assert resumed["best"] == at_once["best"]


def cli(tmp_path, *args):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(examples("train", 32), tmp_path / "train.jsonl")
    write_examples(examples("val", 8), tmp_path / "val.jsonl")
    base = ["--data", str(tmp_path), "--out", str(tmp_path / "out"), "--no-balance"]
    return fit_module.main([*base, "--epochs", "1", "--batch", "8", *args])


def test_cli_resume_keeps_the_dropout_of_the_run(tmp_path, monkeypatch):
    from docta_train import fit as fit_module

    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    assert cli(tmp_path, "--dropout", "0.1") == 0
    assert cli(tmp_path, "--resume") == 0
    saved = torch.load(tmp_path / "out" / "last.pt", weights_only=True)
    assert saved["epoch"] == 2 and saved["config"]["dropout"] == 0.1


def test_cli_refuses_a_negative_warmup_and_an_impossible_dropout(tmp_path):
    import pytest

    for bad in (["--warmup", "-5"], ["--dropout", "1.5"], ["--dropout", "-0.1"]):
        with pytest.raises(SystemExit):
            cli(tmp_path, *bad)
