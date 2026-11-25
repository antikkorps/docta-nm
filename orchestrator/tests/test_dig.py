"""DigModel: keyword, plus a second search driven by what the first one revealed."""

import re

from docta_orchestrator.loop import run
from docta_orchestrator.models.dig import DigModel, revealed
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Exit, Framing

LINK = Hit("s1", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.9)
FOUNDING = Hit("s2", "Annals of Varnelle", "Varnelle was founded in 1432.", 0.9)
GUILD = Hit("s3", "Notes on Brask", "Brask has 300 members. Brask keeps its seat in Kel Dorn.", 0.8)
HITS = [LINK, FOUNDING, GUILD]
CHAIN = "In what year was the city where Orlan Teste lives founded?"


def top_hit(query):
    """Like the real tool with a tiny k: the sources sharing most words, ties
    all kept (a bridge's name is once in its link and once in its value)."""
    words = set(re.findall(r"\w+", query.lower()))
    scored = [(len(words & set(re.findall(r"\w+", h.body.lower()))), h) for h in HITS]
    best = max(n for n, _ in scored)
    return [h for n, h in scored if n == best] if best else []


def ask(question):
    return run(question, DigModel(), top_hit)


def test_reveals_the_names_a_sentence_about_the_subject_brings():
    assert revealed(CHAIN, (LINK, FOUNDING)) == ["Varnelle"]
    assert revealed("In what year was Brask founded?", (GUILD,)) == ["Kel Dorn"]


def test_digs_into_what_the_first_round_revealed_by_its_name():
    # The question's other words ("where", "city", "lives") match sources
    # about anything near the bridge and crowd out the one stating its value.
    assert DigModel().plan(Framing(CHAIN), (LINK,), round_=1) == ["Varnelle"]


def test_answers_a_chain_citing_every_link():
    outcome, trace = ask(CHAIN)
    assert (outcome.answer, outcome.cited) == ("1432", ("s1", "s2"))
    assert [e["round"] for e in trace if e["state"] == "plan"] == [1, 2]


def test_a_direct_question_is_answered_like_keyword_in_one_round():
    outcome, trace = ask("In what year was Varnelle founded?")
    assert outcome.answer == "1432"
    assert sum(e["state"] == "plan" for e in trace) == 1


def test_never_digs_past_a_direct_question_it_cannot_answer():
    # Orlan's birth year is in no source: following Varnelle would "find" 1432.
    outcome, _ = ask("In what year was Orlan Teste born?")
    assert outcome.exit == Exit.NOT_FOUND


def test_follows_the_link_whose_answer_fits_the_question():
    # Orlan's guild is revealed too; its member count must not answer a founding year.
    guild_link = Hit("s4", "Notes on Orlan Teste", "Orlan Teste belongs to Fini.", 0.9)
    guild = Hit("s5", "Rolls of Fini", "Fini has 300 members.", 0.9)
    found = DigModel().answer(Framing(CHAIN), (guild_link, guild, LINK, FOUNDING))
    assert found == ("1432", ("s1", "s2"))


def test_a_revealed_name_matches_whole_not_by_one_of_its_words():
    link = Hit("s1", "Notes on Orlan Teste", "Orlan Teste lives in Kel Dorn.", 0.9)
    other = Hit("s2", "Annals of Kel Mora", "Kel Mora was founded in 900.", 0.9)
    assert DigModel().answer(Framing(CHAIN), (link, other)) is None


def test_a_fronted_place_is_revealed_without_its_preposition():
    fronted = Hit("s4", "Notes on Orlan Teste", "In Varnelle, Orlan Teste has a home.", 0.9)
    assert revealed(CHAIN, (fronted,)) == ["Varnelle"]
    assert DigModel().answer(Framing(CHAIN), (fronted, FOUNDING)) == ("1432", ("s4", "s2"))


def test_a_sentence_opening_word_is_never_a_revealed_name():
    rolls = Hit("s5", "Notes on Brask", "On the rolls of Brask are 300 members.", 0.8)
    assert revealed("In which city is Brask based?", (rolls,)) == []
