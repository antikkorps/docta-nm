"""The hybrid model: keyword's plan and conclusion, the trained core's judgement."""

from dataclasses import dataclass

from docta_orchestrator.loop import run
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit
from docta_train.core_model import CoreEvaluateModel

S1 = Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432.", 0.9)


def search(query):
    return [S1]


@dataclass
class Stub:
    """Stands in for the trained core: always the same probability."""

    p: float

    def prob_enough(self, question, excerpts):
        return self.p


def test_abstains_when_the_core_judges_the_reading_insufficient():
    outcome, _ = run("In what year was Varnelle founded?", CoreEvaluateModel(Stub(0.2)), search)
    assert outcome.exit == Exit.NOT_FOUND


def test_answers_when_the_core_judges_it_enough():
    outcome, _ = run("In what year was Varnelle founded?", CoreEvaluateModel(Stub(0.9)), search)
    assert (outcome.answer, outcome.cited) == ("1432", ("s1",))


def test_never_answers_without_support_even_if_the_core_says_enough():
    outcome, _ = run("In which city does Orlan Teste live?", CoreEvaluateModel(Stub(0.9)), search)
    assert outcome.exit == Exit.NOT_FOUND


def test_run_core_needs_a_database(monkeypatch, tmp_path):
    from docta_train.run_core import main

    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert main([str(tmp_path / "test.jsonl")]) == 2
