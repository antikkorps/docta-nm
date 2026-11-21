"""core_chain: the orchestrator splits chains, the trained core judges each hop."""

from dataclasses import dataclass, field

from docta_orchestrator.loop import run
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit
from docta_train.core_model import CoreChainModel

LIVES = Hit("s1", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.9)
FOUNDED = Hit("s2", "Annals of Varnelle", "Varnelle was founded in 1432.", 0.9)
CHAIN = "In what year was the city where Orlan Teste lives founded?"
FIRST = "In which city does Orlan Teste live?"
SECOND = "In what year was Varnelle founded?"


@dataclass
class Core:
    """Stands in for the trained core: 'enough' only for the questions listed."""

    enough: set[str]
    calls: list = field(default_factory=list)

    def prob_enough(self, question, excerpts):
        self.calls.append((question, [h.source_id for h in excerpts]))
        return 0.9 if question in self.enough else 0.1


def ask(question, core, hits=(LIVES, FOUNDED)):
    return run(question, CoreChainModel(core), lambda q: list(hits))


def test_a_chain_is_enough_when_both_hops_are():
    outcome, _ = ask(CHAIN, Core({FIRST, SECOND}))
    assert (outcome.answer, set(outcome.cited)) == ("1432", {"s1", "s2"})


def test_the_first_hop_is_judged_on_the_source_revealing_the_bridge_only():
    core = Core({FIRST, SECOND})
    ask(CHAIN, core)
    assert (FIRST, ["s1"]) in core.calls
    assert (SECOND, ["s1", "s2"]) in core.calls


def test_a_missing_hop_means_not_enough():
    for enough in ({FIRST}, {SECOND}, {CHAIN}):
        outcome, _ = ask(CHAIN, Core(enough))
        assert outcome.exit == Exit.NOT_FOUND, enough


def test_a_one_hop_question_is_judged_whole():
    core = Core({SECOND})
    outcome, _ = ask(SECOND, core, hits=(FOUNDED,))
    assert outcome.answer == "1432"
    assert core.calls == [(SECOND, ["s2"])]


def test_the_bridge_found_in_the_first_search_leads_the_second():
    searched = []

    def search(query):
        searched.append(query)
        return [LIVES] if len(searched) == 1 else [FOUNDED]

    outcome, _ = run(CHAIN, CoreChainModel(Core({FIRST, SECOND})), search)
    assert outcome.answer == "1432" and len(searched) == 2
