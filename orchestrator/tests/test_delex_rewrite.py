"""The delexicaliser: real sentences keep their structure, every name and
value becomes an invented one, the same way in the text and in the facts."""

import random

from docta_orchestrator.delex import numbers, rewrite, words

ARTICLE = {
    "qid": "Q1",
    "title": "Mount Brask",
    "links": ["Varn Range", "Orlo Smith"],
    "sections": [
        {
            "path": ["Mount Brask"],
            "paragraphs": [
                "Mount Brask rises 4,808 m in the Varn Range. Orlo Smith climbed it on "
                "August 8, 1786, the 1st ascent. About 40% of climbers turn back.",
                "Brask is a mountain; the range has older peaks.",
            ],
        }
    ],
}
ENTITY = {
    "qid": "Q1",
    "kind": "mountain",
    "label": "Mount Brask",
    "title": "Mount Brask",
    "aliases": ["Brask"],
    "facts": {"elevation_m": [4808], "range": [{"qid": "Q2", "label": "Varn Range"}]},
}


def test_entity_words_skip_any_word_seen_in_lowercase():
    found = words.entity_words([ARTICLE], [ENTITY])
    # "Range" stays a word: "range" appears in lowercase, so it is no name.
    assert found == {"Brask", "Mount", "Orlo", "Smith", "Varn"}


def test_names_are_consistent_skip_corpus_words_and_follow_a_fixed_order():
    namer = words.Namer(["Mountain", "Kelda", "Rosta", "Thive"], forbidden={"mountain"})
    assert namer.assign({"Varn", "Brask"}) == {"Brask": "Kelda", "Varn": "Rosta"}


def test_years_shift_and_quantities_scale_keeping_their_form():
    text = "Rises 4,808 m; climbed on August 8, 1786, the 1st ascent; 40% turn back; 12.5 km."
    out = numbers.rewrite(text, shift=-100, factor=1.5)
    assert (
        out == "Rises 7,212 m; climbed on August 8, 1686, the 1st ascent; 40% turn back; 18.8 km."
    )


def test_decades_shift_and_days_before_a_month_stay():
    out = numbers.rewrite("In the 1990s, on 28 August 1939, in her 20s.", shift=-100, factor=1.5)
    assert out == "In the 1890s, on 28 August 1839, in her 20s."


def test_fact_values_follow_the_same_rules_as_the_text():
    assert numbers.year(1786, shift=-100) == 1686
    assert numbers.quantity(4808, factor=1.5) == 7212
    assert numbers.quantity(12.5, factor=1.5) == 18.8


def test_an_article_and_its_entity_are_rewritten_alike():
    names = {"Brask": "Kelda", "Varn": "Rosta", "Orlo": "Thive", "Smith": "Dorn", "Mount": "Mount"}
    art, ent = rewrite.pair(ARTICLE, ENTITY, names, rng=random.Random(0))
    text = " ".join(art["sections"][0]["paragraphs"])
    assert "Brask" not in text and "Varn" not in text and "Smith" not in text
    assert ent["label"] == "Mount Kelda" and ent["aliases"] == ["Kelda"]
    assert ent["facts"]["range"] == [{"qid": "Q2", "label": "Rosta Range"}]
    assert str(ent["facts"]["elevation_m"][0]) in text.replace(",", "")
    assert art["links"] == ["Rosta Range", "Thive Dorn"]


def test_rewriting_is_deterministic():
    names = {"Brask": "Kelda", "Varn": "Rosta", "Orlo": "Thive", "Smith": "Dorn"}
    a = rewrite.pair(ARTICLE, ENTITY, names, rng=random.Random(3))
    b = rewrite.pair(ARTICLE, ENTITY, names, rng=random.Random(3))
    assert a == b
