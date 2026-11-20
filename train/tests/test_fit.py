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
    """Enough exactly when the excerpt states a founding year: learnable. One
    world each, so the template's words reach the vocabulary's world count."""
    return [
        Example(
            f"w{i}",
            f"q{i}",
            split,
            "When was Brask founded?",
            (YES if i % 2 else NO,),
            bool(i % 2),
        )  # fmt: skip
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
    monkeypatch.setattr(
        fit_module, "fit", lambda train, val, *a, **kw: seen.update(val=len(val)) or {}
    )
    fit_module.main(["--data", str(tmp_path), "--max-val", "3", "--no-balance"])
    assert seen["val"] == 3


def shaped(split, n):
    """One shape is never enough, the other half the time: the shape gives
    away three labels in four."""
    return [
        Example(f"w{i}", f"q{i}", split,
                "When was Brask founded?" if i % 2 else "Where does Ana Lo live?",
                (YES,), i % 4 == 3)
        for i in range(n)
    ]  # fmt: skip


def run_cli(tmp_path, monkeypatch, *flags):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    write_examples(shaped("train", 16), tmp_path / "train.jsonl")
    write_examples(shaped("val", 16), tmp_path / "val.jsonl")
    seen = {}
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(
        fit_module,
        "fit",
        lambda train, val, *a, **kw: seen.update(train=train, val=val, **kw) or {},
    )
    fit_module.main(["--data", str(tmp_path), *flags])
    return seen


def test_cli_balances_labels_within_each_shape(tmp_path, monkeypatch):
    seen = run_cli(tmp_path, monkeypatch)
    for split in ("train", "val"):
        labels = [e.label for e in seen[split]]
        assert labels.count(True) == labels.count(False) == 4  # the chain shape went


def test_cli_can_keep_every_example(tmp_path, monkeypatch):
    seen = run_cli(tmp_path, monkeypatch, "--no-balance")
    assert len(seen["train"]) == len(seen["val"]) == 16


def test_cli_reports_what_the_shape_alone_would_score(tmp_path, monkeypatch, capsys):
    run_cli(tmp_path, monkeypatch)
    first, rest = capsys.readouterr().err.split("\n", 1)
    assert first == "shape alone on val: 0.750 unbalanced, 0.500 as trained on (8 train, 8 val)"
    assert json.loads(rest)["shape_accuracy"] == {"unbalanced": 0.75, "trained_on": 0.5}


def test_cli_caps_before_balancing(tmp_path, monkeypatch):
    seen = run_cli(tmp_path, monkeypatch, "--max-train", "8")
    labels = [e.label for e in seen["train"]]
    assert labels.count(True) == labels.count(False) == 2


def test_cli_stops_when_balancing_leaves_nothing(tmp_path, monkeypatch, capsys):
    from docta_train import fit as fit_module
    from docta_train.evaluate_data import write_examples

    one_label = [e for e in shaped("train", 16) if not e.label]
    write_examples(one_label, tmp_path / "train.jsonl")
    write_examples(shaped("val", 16), tmp_path / "val.jsonl")
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(fit_module, "fit", lambda *a: pytest.fail("trained on nothing"))
    assert fit_module.main(["--data", str(tmp_path)]) == 1
    assert "no example" in capsys.readouterr().err


def test_predictor_says_how_to_fix_a_checkpoint_from_before_word_tokens(tmp_path):
    import torch

    torch.save({"config": TINY.__dict__, "state": {}, "metrics": {}}, tmp_path / "old.pt")
    with pytest.raises(ValueError, match="rerun `just fit`"):
        Predictor(tmp_path / "old.pt")


def test_the_checkpoint_carries_the_vocabulary_learned_from_train(tmp_path):
    import torch

    fit(examples("train", 32), examples("val", 8), tmp_path, TINY, epochs=1, batch=8,
        lr=3e-3, seed=0)  # fmt: skip
    words = torch.load(tmp_path / "model.pt", weights_only=True)["vocab"]
    assert {"<cls>", "<name_1>", "when", "founded", "?"} <= set(words)


def test_predictor_tokenizes_with_the_vocabulary_it_was_trained_with(tmp_path):
    import torch

    fit(examples("train", 32), examples("val", 8), tmp_path, TINY, epochs=1, batch=8,
        lr=3e-3, seed=0)  # fmt: skip
    saved = torch.load(tmp_path / "model.pt", weights_only=True)["vocab"]
    assert Predictor(tmp_path / "model.pt").vocab.words == tuple(saved)


def test_cli_learns_the_vocabulary_before_balancing(tmp_path, monkeypatch):
    # Balancing picks examples by label: building after it would let labels
    # decide which words make the vocabulary.
    from docta_train.words import Vocab

    built_from = []
    real = Vocab.build.__func__
    monkeypatch.setattr(
        Vocab,
        "build",
        classmethod(
            lambda cls, ex, *a: built_from.append(len(ex := list(ex))) or real(cls, ex, *a)
        ),
    )
    seen = run_cli(tmp_path, monkeypatch)
    assert built_from == [16] and len(seen["train"]) == 8
