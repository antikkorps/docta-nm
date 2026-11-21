"""A human hint mid-way: it may steer the search, never support the answer."""

from dataclasses import dataclass, field

from docta_orchestrator.human import human_hint
from docta_orchestrator.loop import run
from docta_orchestrator.models import DigModel
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Conclusion, Confrontation, Exit, Framing, Verdict

CHAIN = "In what year was the city where Orlan Teste lives founded?"
LIVES = Hit("s1", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.9)
FOUNDED = Hit("s2", "Annals of Varnelle", "Varnelle was founded in 1432.", 0.9)
BRASK = Hit("s3", "Annals of Brask", "Brask was founded in 1610.", 0.9)
RIGHT = human_hint("Orlan Teste lives in Varnelle.")
WRONG = human_hint("Orlan Teste lives in Brask.")


def search(query):
    """The first search misses the link source: only a hint can lead on."""
    if "Varnelle" in query:
        return [FOUNDED, LIVES]
    if "Brask" in query:
        return [BRASK]
    return []


@dataclass
class Recording:
    seen: dict = field(default_factory=dict)

    def frame(self, question):
        return Framing(question, ())

    def plan(self, framing, evidence, round_):
        self.seen[f"plan{round_}"] = [h.source_id for h in evidence]
        return [framing.question] if round_ < 2 else []

    def evaluate(self, framing, evidence):
        self.seen.setdefault("evaluate", []).append([h.source_id for h in evidence])
        return Verdict(False)

    def confront(self, framing, evidence):
        return Confrontation(())

    def conclude(self, framing, evidence, confrontation):
        return Conclusion(None, (), None)


def test_a_hint_reaches_plan_from_the_second_round_only():
    model = Recording()
    run(CHAIN, model, lambda q: [LIVES], hint=RIGHT)
    assert "human" not in model.seen["plan0"] and "human" in model.seen["plan1"]


def test_evaluate_never_reads_the_hint():
    model = Recording()
    run(CHAIN, model, lambda q: [LIVES], hint=RIGHT)
    assert all("human" not in read for read in model.seen["evaluate"])


def test_the_trace_shows_when_the_human_spoke():
    _, trace = run(CHAIN, Recording(), lambda q: [LIVES], hint=RIGHT)
    said = [e for e in trace if e["state"] == "human"]
    assert said == [{"state": "human", "round": 2, "hint": "Orlan Teste lives in Varnelle."}]


def test_a_right_hint_leads_to_an_answer_the_sources_support():
    assert run(CHAIN, DigModel(), search)[0].exit == Exit.NOT_FOUND  # alone, stuck
    outcome, _ = run(CHAIN, DigModel(), search, hint=RIGHT)
    assert (outcome.answer, set(outcome.cited)) == ("1432", {"s1", "s2"})


def test_a_wrong_hint_steers_the_search_but_supports_nothing():
    outcome, trace = run(CHAIN, DigModel(), search, hint=WRONG)
    assert any("Brask" in q for e in trace if e["state"] == "plan" for q in e["queries"])
    assert outcome.answer != "1610" and outcome.exit == Exit.NOT_FOUND
