"""Which models a run of the trained cores scores."""

import docta_train.run_core as run_core


def models_for(tmp_path, monkeypatch, reader_exists, *extra):
    monkeypatch.setattr(run_core, "Predictor", lambda path: object())
    monkeypatch.setattr(run_core, "ReadPredictor", lambda path: object())
    reader = tmp_path / "read.pt"
    if reader_exists:
        reader.touch()
    args = run_core.parse_args(["data/test.jsonl", "--reader", str(reader),
                                "--out", str(tmp_path), *extra])  # fmt: skip
    return [m.name for m in run_core.build_models(args)]


def test_a_read_core_brings_core_read_and_core_bridge(tmp_path, monkeypatch):
    names = models_for(tmp_path, monkeypatch, True)
    assert names == ["core_evaluate", "core_chain", "core_read", "core_bridge"]


def test_without_a_read_core_its_models_old_predictions_go(tmp_path, monkeypatch):
    for name in ("core_read", "core_bridge"):
        (tmp_path / f"{name}.jsonl").write_text("{}\n")
    assert models_for(tmp_path, monkeypatch, False) == ["core_evaluate", "core_chain"]
    assert not list(tmp_path.glob("core_*.jsonl"))
