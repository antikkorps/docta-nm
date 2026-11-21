"""The three phase-0 fake models, run through the real loop."""

import re

from docta_orchestrator.loop import run
from docta_orchestrator.models import AlwaysAbstainModel, KeywordModel, RandomModel
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit

HITS = [
    Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432. Brask has 300 members.", 0.9),
    Hit("s2", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.7),
    Hit("s3", "Annals of Brask", "Brask was founded in 1610.", 0.8),
]


def fake_search(query):
    """OR over words, ranked by how many words match, like the real tool."""
    words = set(re.findall(r"\w+", query.lower()))
    scored = [(len(words & set(re.findall(r"\w+", h.body.lower()))), h) for h in HITS]
    return [h for n, h in sorted(scored, key=lambda x: -x[0]) if n]


def ask(model, question):
    return run(question, model, fake_search)[0]


def test_always_abstain_never_searches():
    outcome, trace = run("In what year was Varnelle founded?", AlwaysAbstainModel(), fake_search)
    assert outcome.exit == Exit.NOT_FOUND
    assert "consult" not in [e["state"] for e in trace]


def test_random_model_is_reproducible_per_question():
    questions = [f"In what year was Varnelle founded, take {i}?" for i in range(20)]
    first = [ask(RandomModel(seed=1), q) for q in questions]
    assert first == [ask(RandomModel(seed=1), q) for q in reversed(questions)][::-1]
    assert first != [ask(RandomModel(seed=2), q) for q in questions]
    assert any(o.exit for o in first) and any(o.answer for o in first)


def test_keyword_model_reads_a_number_or_a_name_next_to_the_subject():
    year = ask(KeywordModel(), "In what year was Varnelle founded?")
    assert (year.answer, year.cited) == ("1432", ("s1",))
    city = ask(KeywordModel(), "In which city does Orlan Teste live?")
    assert (city.answer, city.cited) == ("Varnelle", ("s2",))


def test_keyword_model_compares_two_numbers():
    older = ask(KeywordModel(), "Which was founded earlier, Brask or Varnelle?")
    assert older.answer == "Varnelle"
    assert set(older.cited) == {"s1", "s3"}  # Varnelle in s1, Brask in s3


def test_keyword_model_asks_a_name_only_with_which_or_who():
    # A rule on what a question asks, not on its opening words: any other
    # wording of a number question still reads a number.
    year = ask(KeywordModel(), "When, as a year, was Varnelle founded?")
    assert (year.answer, year.cited) == ("1432", ("s1",))
    city = ask(KeywordModel(), "Name the city which Orlan Teste lives in?")
    assert city.answer == "Varnelle"


def test_keyword_model_compares_two_names_joined_by_or_in_any_wording():
    older = ask(KeywordModel(), "Which of Brask or Varnelle was founded earlier?")
    assert older.answer == "Varnelle"
    # A question's opening word is never a name, capitalised or not.
    later = ask(KeywordModel(), "Of Brask or Varnelle, which was founded more recently?")
    assert later.answer == "Brask"


def test_keyword_model_compares_two_people_with_two_word_names():
    people = [
        Hit("p1", "Notes", "Orlan Teste is 182 centimetres tall.", 0.9),
        Hit("p2", "Notes", "Mira Kol is 175 centimetres tall.", 0.9),
    ]
    question = "Which of Mira Kol or Orlan Teste is taller?"
    outcome, _ = run(question, KeywordModel(), lambda q: people)
    assert (outcome.answer, set(outcome.cited)) == ("Orlan Teste", {"p1", "p2"})


def test_keyword_model_abstains_when_nothing_relevant_comes_back():
    assert ask(KeywordModel(), "In what year was Orlan Teste born?").exit == Exit.NOT_FOUND
    assert ask(KeywordModel(), "How tall is Zorvath, in centimetres?").exit == Exit.NOT_FOUND


def test_keyword_model_never_answers_a_sentence_opening_word():
    from docta_orchestrator.models import KeywordModel
    from docta_orchestrator.search import Hit
    from docta_orchestrator.states import Framing

    rolls = Hit("s1", "Survey", "On the rolls of Kortha are 40 members.", 0.9)
    seat = Hit("s2", "Notes", "Kortha keeps its seat in Varnelle.", 0.9)
    found = KeywordModel().answer(Framing("Which city holds the seat of Kortha?"), (rolls, seat))
    assert found == ("Varnelle", ("s2",))


def test_a_frame_word_opening_a_sentence_is_never_a_name():
    # Any capitalised word of a seen template is a template word, not a name.
    from docta_orchestrator.models import KeywordModel
    from docta_orchestrator.search import Hit
    from docta_orchestrator.states import Framing

    said = Hit("s1", "Notes", "It is said that Orlan Teste belongs to Kortha.", 0.9)
    records = Hit("s2", "Notes", "Records show that Mira Kol lives in Varnelle.", 0.9)
    model = KeywordModel()
    assert model.answer(Framing("Which guild does Orlan Teste belong to?"), (said,)) == (
        "Kortha",
        ("s1",),
    )
    assert model.answer(Framing("In which city does Mira Kol live?"), (records,))[0] == "Varnelle"
