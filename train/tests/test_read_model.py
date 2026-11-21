"""core_read: core_chain's judgement, the Read core's answers."""

from dataclasses import dataclass

from docta_orchestrator.loop import run
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit
from docta_train.read_model import CoreReadModel

LIVES = Hit("s1", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.9)
FOUNDED = Hit("s2", "Annals of Varnelle", "Varnelle was founded in 1432.", 0.9)
BRASK = Hit("s3", "Annals of Brask", "Brask was founded in 1610.", 0.9)
CHAIN = "In what year was the city where Orlan Teste lives founded?"
FIRST = "In which city does Orlan Teste live?"
SECOND = "In what year was Varnelle founded?"
COMPARE = "Which was founded earlier, Brask or Varnelle?"


@dataclass
class Core:
    """Judges everything enough."""

    def prob_enough(self, question, excerpts):
        return 0.9


@dataclass
class Reader:
    """Reads the listed answers, when their source was read."""

    answers: dict

    def read(self, question, excerpts):
        found = self.answers.get(question)
        read = {h.source_id for h in excerpts}
        return found if found and found[1] in read else None


READER = Reader({
    SECOND: ("1432", "s2"),
    FIRST: ("Varnelle", "s1"),
    f"{COMPARE} Brask": ("1610", "s3"),
    f"{COMPARE} Varnelle": ("1432", "s2"),
})  # fmt: skip


def ask(question, hits=(LIVES, FOUNDED, BRASK), reader=READER):
    return run(question, CoreReadModel(Core(), reader), lambda q: list(hits))[0]


def test_a_one_hop_answer_is_what_the_reader_points_at():
    outcome = ask(SECOND)
    assert (outcome.answer, outcome.cited) == ("1432", ("s2",))


def test_a_comparison_reads_each_value_and_code_compares():
    outcome = ask(COMPARE)
    assert (outcome.answer, set(outcome.cited)) == ("Varnelle", {"s2", "s3"})


def test_a_chain_reads_the_bridge_then_the_value_about_it():
    outcome = ask(CHAIN)
    assert (outcome.answer, set(outcome.cited)) == ("1432", {"s1", "s2"})


def test_the_bridge_the_reader_found_leads_the_second_search():
    searched = []

    def search(query):
        searched.append(query)
        return [LIVES] if len(searched) == 1 else [FOUNDED]

    outcome, _ = run(CHAIN, CoreReadModel(Core(), READER), search)
    assert outcome.answer == "1432" and "Varnelle" in searched[1]


def test_nothing_read_means_not_found():
    assert ask(SECOND, reader=Reader({})).exit == Exit.NOT_FOUND
    assert ask(COMPARE, hits=(BRASK,)).exit == Exit.NOT_FOUND  # one value only


def test_a_bridge_is_a_name_a_sentence_about_the_subject_brings_in():
    for misread in [("lives", "s1"), ("Orlan Teste", "s1"), ("1432", "s2")]:
        # Whatever the bridge, a value about it would be read: only the bridge check stops it.
        second = f"In what year was {misread[0]} founded?"
        reader = Reader({**READER.answers, FIRST: misread, second: ("1432", "s2")})
        assert ask(CHAIN, reader=reader).exit == Exit.NOT_FOUND


def test_the_first_hop_is_judged_on_the_sentence_naming_the_bridge():
    # On the whole source, a guild named next to the city would pass for it.
    both = Hit("s1", "Notes", "Orlan Teste lives in Varnelle. Orlan Teste is in Kel Guild.", 0.9)
    judged = []

    @dataclass
    class Recording:
        def prob_enough(self, question, excerpts):
            judged.append((question, [h.body for h in excerpts]))
            return 0.9

    reader = Reader({**READER.answers, FIRST: ("Kel Guild", "s1")})
    run(CHAIN, CoreReadModel(Recording(), reader), lambda q: [both, FOUNDED])
    assert (FIRST, ["Orlan Teste is in Kel Guild."]) in judged


def test_a_one_hop_answer_is_a_number_or_a_name():
    assert ask(SECOND, reader=Reader({SECOND: ("founded", "s2")})).exit == Exit.NOT_FOUND
