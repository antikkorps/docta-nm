"""core_bridge: no rule splits a chain; the Read core points at what to search next."""

from dataclasses import dataclass

from docta_orchestrator.human import human_hint
from docta_orchestrator.loop import run
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit, Framing
from docta_train.bridge_model import CoreBridgeModel

LIVES = Hit("s1", "Notes on Orlan Teste", "Orlan Teste resides in Varnelle.", 0.9)
FOUNDED = Hit("s2", "Annals of Varnelle", "Varnelle was founded in 1432.", 0.9)
CHAIN = "In what year was the city Orlan Teste resides in founded?"  # no seen link wording


@dataclass
class Core:
    """Enough once the founding was read."""

    def prob_enough(self, question, excerpts):
        return 0.9 if any(h.source_id == "s2" for h in excerpts) else 0.1


@dataclass
class Reader:
    """Two pointers: the answer once read; the bridge, wherever it is said."""

    def read(self, question, excerpts):
        return ("1432", "s2") if any(h.source_id == "s2" for h in excerpts) else None

    def bridge(self, question, excerpts):
        said = [h.source_id for h in excerpts if "Varnelle" in h.body]
        return ("Varnelle", said[0]) if said else None


def search_by(index):
    searched = []

    def search(query):
        searched.append(query)
        return list(index.get(query, ()))

    return search, searched


def ask(index, hint=None):
    search, searched = search_by(index)
    outcome, _ = run(CHAIN, CoreBridgeModel(Core(), Reader()), search, hint=hint)
    return outcome, searched


def test_a_chain_searches_the_name_the_core_points_at_then_reads_the_answer():
    outcome, searched = ask({CHAIN: [LIVES], "Varnelle": [FOUNDED]})
    assert searched == [CHAIN, "Varnelle"]
    assert (outcome.answer, outcome.cited) == ("1432", ("s1", "s2"))


def test_nothing_named_to_search_means_not_found():
    outcome, searched = ask({CHAIN: [Hit("s3", "T", "Brask has 300 members.", 0.9)]})
    assert outcome.exit == Exit.NOT_FOUND and searched == [CHAIN]


def test_a_hint_names_what_to_search_and_sources_still_carry_the_answer():
    hint = human_hint("Orlan Teste resides in Varnelle.")
    outcome, searched = ask({CHAIN: [], "Varnelle": [LIVES, FOUNDED]}, hint=hint)
    assert searched[1] == "Varnelle"
    assert (outcome.answer, set(outcome.cited)) == ("1432", {"s1", "s2"})


def test_a_question_in_the_name_is_never_the_next_search():
    reader = Reader()
    model = CoreBridgeModel(Core(), reader)
    asked = "In what year was Varnelle founded?"
    assert model.plan(Framing(asked), (LIVES,), round_=1) == []


def test_a_one_hop_answer_cites_its_own_source_only():
    # The answer is Varnelle, from s1. Set s1 aside and the core points at
    # Varnelle in s2: the answer itself, not a bridge that led to it.
    @dataclass
    class Names:
        def read(self, question, excerpts):
            said = [h.source_id for h in excerpts if "Varnelle" in h.body]
            return ("Varnelle", said[0]) if said else None

    asked = "In which city does Orlan Teste reside?"
    found = CoreBridgeModel(Core(), Names()).answer(Framing(asked), (LIVES, FOUNDED))
    assert found == ("Varnelle", ("s1",))


def test_a_template_word_is_never_a_name_to_search():
    @dataclass
    class Opening:
        def bridge(self, question, excerpts):
            return "The", "s1"

    assert CoreBridgeModel(Core(), Opening()).plan(Framing(CHAIN), (LIVES,), round_=1) == []


def test_a_comparison_never_digs():
    asked = "Which was founded earlier, Brask or Kel Dorn?"
    assert CoreBridgeModel(Core(), Reader()).plan(Framing(asked), (LIVES,), round_=1) == []


def test_a_wrong_hint_never_carries_an_answer_no_source_ties_to_the_subject():
    # The hint sends the dig to Brask; Brask's founding is read, but no source
    # says Orlan Teste lives there, so that year answers nothing asked.
    @dataclass
    class Hinted:
        def read(self, question, excerpts):
            return ("1100", "s4") if any(h.source_id == "s4" for h in excerpts) else None

        def bridge(self, question, excerpts):
            said = [h.source_id for h in excerpts if "Brask" in h.body]
            return ("Brask", said[0]) if said else None

    brask = Hit("s4", "Annals of Brask", "Brask was founded in 1100.", 0.9)
    search, _ = search_by({CHAIN: [LIVES], "Brask": [brask]})
    hint = human_hint("Orlan Teste resides in Brask.")
    outcome, _ = run(CHAIN, CoreBridgeModel(AnyFounding(), Hinted()), search, hint=hint)
    assert outcome.answer is None


@dataclass
class AnyFounding:
    """Enough once any founding was read."""

    def prob_enough(self, question, excerpts):
        return 0.9 if any("founded" in h.body for h in excerpts) else 0.1


def test_a_link_and_answer_in_one_source_cite_it_once():
    both = Hit("s1", "Notes", "Orlan Teste resides in Varnelle. Varnelle was founded in 1432.", 0.9)

    @dataclass
    class Year:
        def read(self, question, excerpts):
            return ("1432", "s1") if excerpts else None

    found = CoreBridgeModel(Core(), Year()).answer(Framing(CHAIN), (both,))
    assert found == ("1432", ("s1",))


def test_the_link_cited_is_the_source_tying_the_subject_to_the_bridge():
    # The core may point at Varnelle where its population is said: the link
    # cited is still the sentence saying Orlan Teste resides there.
    people = Hit("s3", "Census", "Varnelle has 300 inhabitants.", 0.9)
    found = CoreBridgeModel(Core(), Reader()).answer(Framing(CHAIN), (FOUNDED, people, LIVES))
    assert found == ("1432", ("s1", "s2"))
