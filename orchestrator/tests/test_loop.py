"""The state machine, driven by a scripted model and an in-memory search."""

import json
from dataclasses import dataclass, field

import pytest
from docta_orchestrator.loop import run
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Conclusion, Confrontation, Exit, Framing, Verdict

S1 = Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432.", 0.9)
S2 = Hit("s2", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.7)
INDEX = {"Varnelle": [S1, S2], "Orlan": [S2]}


def fake_search(query: str) -> list[Hit]:
    return INDEX.get(query, [])


@dataclass
class Scripted:
    """Answers each state from a script; records what it was asked."""

    name: str = "scripted"
    framing: Framing = Framing("restated", ("year",))
    queries: list[list[str]] = field(default_factory=lambda: [["Orlan"], ["Varnelle"]])
    enough_after: int = 2  # rounds of search needed before "enough"
    verdict_exit: Exit | None = None
    confrontation: Confrontation = Confrontation(("agree",))
    conclusion: Conclusion = Conclusion("1432", ("s2", "s1"), 0.8)
    rounds_planned: int = 0

    def frame(self, question):
        return self.framing

    def plan(self, framing, evidence, round_):
        self.rounds_planned += 1
        return self.queries[round_] if round_ < len(self.queries) else []

    def evaluate(self, framing, evidence):
        return Verdict(self.rounds_planned >= self.enough_after, self.verdict_exit)

    def confront(self, framing, evidence):
        return self.confrontation

    def conclude(self, framing, evidence, confrontation):
        return self.conclusion


def states(trace):
    return [event["state"] for event in trace]


def test_full_path_loops_back_to_plan_until_enough():
    outcome, trace = run("In what year...?", Scripted(), fake_search)
    assert outcome.answer == "1432" and outcome.exit is None
    assert outcome.cited == ("s2", "s1")
    first_round = ["plan", "consult", "evaluate"]
    assert states(trace) == ["frame", *first_round, *first_round, "confront", "conclude"]


def test_consult_runs_the_planned_queries_and_dedupes_excerpts():
    _, trace = run("q", Scripted(), fake_search)
    consults = [e for e in trace if e["state"] == "consult"]
    assert consults[0]["queries"] == ["Orlan"]
    assert [x["source_id"] for x in consults[0]["excerpts"]] == ["s2"]
    assert [x["source_id"] for x in consults[1]["excerpts"]] == ["s1"]  # s2 already seen


def test_gives_up_with_not_found_after_max_rounds():
    model = Scripted(queries=[["nothing"]] * 5, enough_after=99)
    outcome, trace = run("q", model, fake_search, max_rounds=3)
    assert outcome.exit == Exit.NOT_FOUND and outcome.answer is None
    assert model.rounds_planned == 3
    assert trace[-1] == {"state": "evaluate", "enough": False, "exit": "not_found"}


def test_emergency_exits_stop_the_loop():
    ill = Scripted(framing=Framing("?", exit=Exit.ILL_POSED))
    assert run("q", ill, fake_search)[0].exit == Exit.ILL_POSED
    assert states(run("q", ill, fake_search)[1]) == ["frame"]
    lost = Scripted(verdict_exit=Exit.NOT_FOUND)
    assert run("q", lost, fake_search)[0].exit == Exit.NOT_FOUND
    torn = Scripted(confrontation=Confrontation(exit=Exit.CONTRADICTION))
    assert run("q", torn, fake_search)[0].exit == Exit.CONTRADICTION


def test_a_plan_without_queries_skips_consult():
    _, trace = run("q", Scripted(queries=[], enough_after=1), fake_search)
    assert states(trace)[:3] == ["frame", "plan", "evaluate"]


def test_only_consulted_sources_can_be_cited():
    model = Scripted(conclusion=Conclusion("1432", ("s1", "s9", "s2"), 0.8))
    outcome, trace = run("q", model, fake_search)
    assert outcome.cited == ("s1", "s2")
    assert trace[-1]["dropped_citations"] == ["s9"]


def test_trace_is_plain_json():
    _, trace = run("q", Scripted(), fake_search)
    assert json.loads(json.dumps(trace)) == trace


def test_cannot_conclude_without_having_read_anything():
    # "Enough" with no excerpt is not enough: nothing may be stated unsourced.
    model = Scripted(queries=[], enough_after=0)
    outcome, trace = run("q", model, fake_search, max_rounds=2)
    assert outcome.exit == Exit.NOT_FOUND
    assert "conclude" not in states(trace)
    assert model.rounds_planned == 2


def test_rejects_a_round_limit_below_one():
    with pytest.raises(ValueError, match="max_rounds"):
        run("q", Scripted(), fake_search, max_rounds=0)
