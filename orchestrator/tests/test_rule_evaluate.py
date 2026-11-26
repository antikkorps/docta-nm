"""RuleEvaluateModel: a hand-written Evaluate that knows only the seen templates,
the honest floor a trained Evaluate core has to beat."""

from docta_orchestrator.loop import run
from docta_orchestrator.models.rule_evaluate import (
    RuleEvaluateModel,
    needs,
    read_sentence,
    resolved,
    statements,
)
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit

HITS = [
    Hit("s1", "Register of Orlan Teste", "Orlan Teste resides in Varnelle.", 0.9),
    Hit("s2", "Annals of Varnelle", "The founding of Varnelle dates back to 1432.", 0.9),
    Hit("s3", "Notes on Brask", "Brask was founded in 1610. Brask has 300 members.", 0.8),
    Hit("s4", "Notes on Kel Dorn", "Kel Dorn stands 181 cm tall.", 0.8),
]


def ask(question, hits=HITS):
    return run(question, RuleEvaluateModel(), lambda q: hits)[0]


def test_reads_the_facts_a_seen_sentence_states():
    assert read_sentence("Orlan Teste resides in Varnelle.") == [
        ("Orlan Teste", "lives_in", "Varnelle")
    ]
    assert read_sentence("Kel Dorn stands 181 cm tall.") == [("Kel Dorn", "height", "181")]
    assert read_sentence("Varnelle lies by the river.") == []


def test_turns_each_question_shape_into_the_facts_it_needs():
    assert needs("In what year was Brask founded?") == [("Brask", ("founded_in",))]
    assert needs("In what year was the city where Orlan Teste lives founded?") == [
        ("Orlan Teste", ("lives_in", "founded_in"))
    ]
    assert needs("Who is taller, Kel Dorn or Orlan Teste?") == [
        ("Kel Dorn", ("height",)),
        ("Orlan Teste", ("height",)),
    ]
    assert needs("What year was Brask founded in?") is None  # held-out: unknown to the rule


def test_enough_only_when_every_needed_fact_was_read():
    assert ask("In what year was Brask founded?").answer == "1610"
    # Keyword cannot read a chain's answer, so the loop still abstains on it
    # (as for the trained core): check the rule's own judgement directly.
    (chain,) = needs("In what year was the city where Orlan Teste lives founded?")
    assert resolved(chain, statements(tuple(HITS)))
    assert not resolved(chain, statements(tuple(HITS[:1])))


def test_abstains_where_keyword_would_answer_from_the_wrong_fact():
    # Keyword reads "300" next to Brask; the rule knows no population was read.
    assert ask("What is the population of Brask?").exit == Exit.NOT_FOUND


def test_comparison_needs_both_sides():
    assert ask("Who is taller, Kel Dorn or Orlan Teste?").exit == Exit.NOT_FOUND
    taller = Hit("s5", "Notes on Orlan Teste", "Orlan Teste is 175 centimetres tall.", 0.8)
    both = ask("Who is taller, Kel Dorn or Orlan Teste?", [*HITS, taller])
    assert (both.answer, both.exit) == ("Kel Dorn", None)


def test_unknown_phrasing_falls_back_to_keyword():
    assert ask("How tall is Kel Dorn?").answer == "181"


def test_a_pronoun_reads_as_the_subject_of_the_sentence_before():
    from docta_orchestrator.models.rule_evaluate import read_body

    body = "Kel Dorn lives in Brask. She was born in 1950."
    assert read_body(body) == [
        [("Kel Dorn", "lives_in", "Brask")],
        [("Kel Dorn", "born_in", "1950")],
    ]


def test_a_pronoun_with_no_readable_sentence_before_states_nothing():
    from docta_orchestrator.models.rule_evaluate import read_body

    assert read_body("She was born in 1950.") == [[]]
