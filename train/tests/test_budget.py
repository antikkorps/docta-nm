"""--max-minutes is a budget, not a hint: an epoch that would likely end
past it is not started. An epoch is never cut half-way, or --resume could
not go on exactly as if the run had never stopped."""

from types import SimpleNamespace

from docta_train import fit as fit_module
from docta_train import fit_read as fit_read_module
from docta_train.training import over_budget
from test_fit import TINY, examples
from test_fit_read import examples as reads


def test_the_next_epoch_is_skipped_when_it_would_end_past_the_budget():
    assert not over_budget(elapsed=60, last_epoch=60, max_minutes=2.5)
    assert over_budget(elapsed=120, last_epoch=60, max_minutes=2.5)
    assert not over_budget(elapsed=10_000, last_epoch=60, max_minutes=None)


def test_fit_stops_before_an_epoch_that_would_overrun(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(fit_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    real = fit_module.next_epoch

    def one_minute_epoch(*args):
        clock.now += 60
        return real(*args)

    monkeypatch.setattr(fit_module, "next_epoch", one_minute_epoch)
    fit_module.fit(examples("train", 16), examples("val", 4), tmp_path, TINY,
                   epochs=5, batch=8, lr=1e-3, seed=0, max_minutes=2.5)  # fmt: skip
    # After two minutes a third would end at three, past 2.5: not started.
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_fit_read_stops_before_an_epoch_that_would_overrun(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(fit_read_module, "time", SimpleNamespace(monotonic=lambda: clock.now))
    real = fit_read_module.run_epoch

    def one_minute_epoch(*args):
        clock.now += 60
        return real(*args)

    monkeypatch.setattr(fit_read_module, "run_epoch", one_minute_epoch)
    fit_read_module.fit_read(reads("train", 16), reads("val", 4), tmp_path, TINY,
                             epochs=5, batch=8, lr=1e-3, seed=0, max_minutes=2.5)  # fmt: skip
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2


def test_the_cli_reports_missing_data_instead_of_a_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    assert fit_module.main(["--data", str(tmp_path / "nowhere")]) == 1
    assert "just evaluate-data" in capsys.readouterr().err


def disk_full(*args, **kwargs):
    raise OSError("No space left on device")


def test_the_evaluate_cli_reports_an_unwritable_out_instead_of_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(fit_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(fit_module, "load", lambda args: ([], [], 0.5, None))
    monkeypatch.setattr(fit_module, "announce", lambda *a: None)
    monkeypatch.setattr(fit_module, "fit", disk_full)
    assert fit_module.main([]) == 1
    assert "No space left on device" in capsys.readouterr().err


def test_the_read_cli_reports_an_unwritable_out_instead_of_a_traceback(monkeypatch, capsys):
    monkeypatch.setattr(fit_read_module, "protect_machine", lambda **kw: None)
    monkeypatch.setattr(fit_read_module, "read_reads", lambda path: iter([]))
    monkeypatch.setattr(fit_read_module, "fit_read", disk_full)
    assert fit_read_module.main([]) == 1
    assert "No space left on device" in capsys.readouterr().err
