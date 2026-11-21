"""Chains split into one-hop questions the Evaluate core already judges."""

from docta_orchestrator.models.decompose import bridges, split_chain
from docta_orchestrator.search import Hit

LIVES = Hit("s1", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.9)
GUILD = Hit("s2", "Register", "Orlan Teste is a member of the Amber Guild.", 0.9)
FOUNDING = Hit("s3", "Annals of Varnelle", "Varnelle was founded in 1432.", 0.9)


def test_a_chain_names_its_link_and_subject():
    chain = split_chain("In what year was the city where Orlan Teste lives founded?")
    assert (chain.link, chain.subject) == ("lives_in", "Orlan Teste")


def test_the_first_hop_is_the_link_question_train_shows():
    chain = split_chain("What year was the guild that Orlan Teste belongs to founded in?")
    assert chain.first_hop == "Which guild does Orlan Teste belong to?"


def test_the_second_hop_keeps_the_question_with_the_bridge_in_place_of_the_link():
    # The outer wording stays: a held-out phrasing remains held out.
    chain = split_chain("What population does the city where Brask is based have?")
    assert chain.link == "based_in"
    assert chain.second_hop("Varnelle") == "What population does Varnelle have?"


def test_a_question_without_a_link_is_no_chain():
    assert split_chain("In what year was Varnelle founded?") is None
    assert split_chain("Which was founded earlier, Varnelle or Brask?") is None


def test_each_bridge_comes_with_the_sources_that_reveal_it():
    chain = split_chain("In what year was the city where Orlan Teste lives founded?")
    found = bridges(chain, (LIVES, GUILD, FOUNDING))
    assert found == {"Varnelle": (LIVES,), "Amber Guild": (GUILD,)}
