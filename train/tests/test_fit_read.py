"""Training the Read core, and the reader that serves it."""

import json

import pytest
from docta_orchestrator.search import Hit
from docta_train.fit_read import fit_read
from docta_train.predictor import Predictor
from docta_train.read_data import ReadExample
from docta_train.reader import ReadPredictor
from test_fit import TINY

BRASK = Hit("s1", "T", "Brask has 300 members. Brask was founded in 1610.", 0.9)
NOISE = Hit("s2", "T", "Varnelle was founded in 1432.", 0.9)


def examples(split, n):
    """The answer sits in different places: learnable only by reading."""
    out = []
    for i in range(n):
        answer = "1610" if i % 2 else "300"
        question = "When was Brask founded?" if i % 2 else "How many members does Brask have?"
        hits = (BRASK, NOISE) if i % 3 else (NOISE, BRASK)
        out.append(ReadExample(f"w{i}", f"q{i}", split, question, hits, answer, ("s1",)))
    return out


def train(out, epochs=12):
    return fit_read(examples("train", 64), examples("val", 16), out, TINY,
                    epochs=epochs, batch=16, lr=3e-3, seed=0)  # fmt: skip


def test_fit_read_learns_to_point_and_logs_exact_answers(tmp_path):
    report = train(tmp_path)
    log = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert {"epoch", "train_loss", "val_loss", "val_exact", "val_by_shape"} <= log[0].keys()
    assert report["last"]["val_exact"] == 1.0


def test_the_reader_answers_and_cites_from_the_saved_core(tmp_path):
    train(tmp_path)
    reader = ReadPredictor(tmp_path / "model.pt")
    assert reader.read("When was Brask founded?", [NOISE, BRASK]) == ("1610", "s1")


def test_the_reader_reads_nothing_without_an_excerpt(tmp_path):
    train(tmp_path, epochs=1)
    assert ReadPredictor(tmp_path / "model.pt").read("When was Brask founded?", []) is None


def test_each_core_refuses_the_other_ones_checkpoint(tmp_path):
    from docta_train.fit import fit
    from test_fit import examples as judged

    train(tmp_path / "read", epochs=1)
    with pytest.raises(ValueError, match="not an Evaluate core"):
        Predictor(tmp_path / "read" / "model.pt")
    fit(judged("train", 8), judged("val", 4), tmp_path / "eval", TINY, epochs=1, batch=4,
        lr=3e-3, seed=0)  # fmt: skip
    with pytest.raises(ValueError, match="not a Read core"):
        ReadPredictor(tmp_path / "eval" / "model.pt")


def test_the_saved_core_is_the_last_epoch(tmp_path):
    # Val saturates at 100% from the first epoch while ood still moves, so
    # val cannot pick an epoch: the run length is the choice, and it is kept.
    import torch

    report = train(tmp_path, epochs=3)
    saved = torch.load(tmp_path / "model.pt", weights_only=False)
    assert saved["metrics"]["epoch"] == report["last"]["epoch"] == 3


def test_a_val_answer_out_of_reach_does_not_wreck_the_loss(tmp_path):
    import math

    lost = ReadExample("w", "q", "val", "When was Brask founded?", (NOISE,), "1610", ("s1",))
    report = fit_read(examples("train", 16), [*examples("val", 4), lost, lost], tmp_path, TINY,
                      epochs=1, batch=16, lr=3e-3, seed=0)  # fmt: skip
    assert math.isfinite(report["last"]["val_loss"])


def test_the_bridge_head_and_the_answer_head_point_apart_on_the_same_reading(tmp_path):
    # One encoder, two pointers: what answers, and what to search next. The
    # same excerpts teach "1610" to one and "Brask" to the other.
    asked = "In what year was the city where Orlan Teste lives founded?"
    both = [
        ReadExample(f"w{i}", f"q{i}", split, asked, (BRASK,), *target)
        for split, n in (("train", 64), ("val", 8))
        for i in range(n)
        for target in ((("1610", ("s1",))), ("Brask", ("s1",), "bridge"))
    ]  # fmt: skip
    train_set = [e for e in both if e.split == "train"]
    report = fit_read(train_set, [e for e in both if e.split == "val"], tmp_path, TINY,
                      epochs=12, batch=16, lr=3e-3, seed=0)  # fmt: skip
    assert report["last"]["val_exact_bridge"] == report["last"]["val_exact_answer"] == 1.0
    reader = ReadPredictor(tmp_path / "model.pt")
    assert reader.read(asked, [BRASK]) == ("1610", "s1")
    assert reader.bridge(asked, [BRASK]) == ("Brask", "s1")
