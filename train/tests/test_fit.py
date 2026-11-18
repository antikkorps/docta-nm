"""Full training run and the predictor that serves the trained core."""

import json

import pytest
from docta_orchestrator.search import Hit
from docta_train.evaluate_data import Example
from docta_train.fit import fit
from docta_train.model import Config
from docta_train.predictor import Predictor

TINY = Config(d_model=32, layers=1, heads=2, ff=64, max_len=96)
YES = Hit("s1", "T", "Brask was founded in 1610.", 0.9)
NO = Hit("s2", "T", "Brask keeps its seat in Varnelle.", 0.9)


def examples(split, n):
    """Enough exactly when the excerpt states a founding year: learnable."""
    return [
        Example(
            "w", f"q{i}", split, "When was Brask founded?", (YES if i % 2 else NO,), bool(i % 2)
        )
        for i in range(n)
    ]


def test_fit_saves_the_best_checkpoint_and_logs_each_epoch(tmp_path):
    out = tmp_path / "core"
    report = fit(
        examples("train", 64), examples("val", 16), out, TINY, epochs=8, batch=16, lr=3e-3, seed=0
    )
    assert (out / "model.pt").exists()
    log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    assert len(log) == 8 and {"epoch", "train_loss", "val_loss", "val_accuracy"} <= log[0].keys()
    assert report["best"]["val_accuracy"] == 1.0


def test_predictor_serves_the_trained_core(tmp_path):
    fit(
        examples("train", 64),
        examples("val", 16),
        tmp_path,
        TINY,
        epochs=8,
        batch=16,
        lr=3e-3,
        seed=0,
    )
    core = Predictor(tmp_path / "model.pt")
    assert core.prob_enough("When was Brask founded?", [YES]) > 0.5
    assert core.prob_enough("When was Brask founded?", [NO]) < 0.5


def test_fit_is_reproducible(tmp_path):
    a = fit(
        examples("train", 32),
        examples("val", 8),
        tmp_path / "a",
        TINY,
        epochs=2,
        batch=8,
        lr=3e-3,
        seed=1,
    )
    b = fit(
        examples("train", 32),
        examples("val", 8),
        tmp_path / "b",
        TINY,
        epochs=2,
        batch=8,
        lr=3e-3,
        seed=1,
    )
    assert a["best"] == b["best"]


def test_cli_protects_the_machine_before_training(tmp_path, monkeypatch):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(examples("train", 4), tmp_path / "train.jsonl")
    write_examples(examples("val", 2), tmp_path / "val.jsonl")
    calls = []
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: calls.append(kw))
    monkeypatch.setattr(fit_module, "fit", lambda *a, **kw: calls.append("fit") or {})
    fit_module.main(["--data", str(tmp_path), "--max-gb", "3", "--threads", "1"])
    assert calls == [{"max_gb": 3.0, "threads": 1}, "fit"]


def test_cli_caps_the_validation_set(tmp_path, monkeypatch):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(examples("train", 4), tmp_path / "train.jsonl")
    write_examples(examples("val", 10), tmp_path / "val.jsonl")
    seen = {}
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(fit_module, "fit", lambda train, val, *a: seen.update(val=len(val)) or {})
    fit_module.main(["--data", str(tmp_path), "--max-val", "3"])
    assert seen["val"] == 3


@pytest.mark.parametrize(("train", "val", "name"), [(0, 16, "training"), (64, 0, "validation")])
def test_fit_refuses_an_empty_split(tmp_path, train, val, name):
    with pytest.raises(ValueError, match=f"{name} split has no example"):
        fit(
            examples("train", train),
            examples("val", val),
            tmp_path,
            TINY,
            epochs=1,
            batch=16,
            lr=3e-3,
            seed=0,
        )
