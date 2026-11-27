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


def test_a_real_text_folder_runs_on_its_question_texts(tmp_path, monkeypatch):
    import contextlib

    seen = {}
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(run_core, "build_models", lambda args: ["model"])
    monkeypatch.setattr(run_core, "questions_of", lambda split: [{"text": f"from {split.name}"}])
    monkeypatch.setattr(run_core.psycopg, "connect", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(run_core, "run_models", lambda q, *rest: seen.update(questions=q))
    (tmp_path / "cities").mkdir()
    assert run_core.main([str(tmp_path / "cities"), "--out", str(tmp_path / "out")]) == 0
    assert seen["questions"] == [{"text": "from cities"}]


def test_the_human_protocol_refuses_a_real_text_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATABASE_URL", "postgresql://unused")
    monkeypatch.setattr(run_core, "build_models", lambda args: ["model"])
    assert run_core.main([str(tmp_path), "--human", str(tmp_path / "h")]) == 1
    assert "world split" in capsys.readouterr().err
