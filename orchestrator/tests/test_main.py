"""Command-line behaviour of `python -m docta_orchestrator.load`."""

import os

import pytest
from docta_orchestrator.load import main


def test_requires_at_least_one_file():
    with pytest.raises(SystemExit) as exit_:
        main([])
    assert exit_.value.code == 2


def test_requires_database_url(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main([str(tmp_path / "x.jsonl")]) == 2
    assert "DATABASE_URL" in capsys.readouterr().err


def test_a_missing_file_is_reported_without_a_traceback(capsys, tmp_path):
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set: run `just up`, then the tests through `just test`")
    # Fails before any write, inside the load transaction: the database is untouched.
    assert main([str(tmp_path / "missing.jsonl")]) == 1
    assert "missing.jsonl" in capsys.readouterr().err
