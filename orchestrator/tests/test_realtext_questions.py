"""Questions on real text: built from the Wikidata key with the wordings the
cores know, and judged answerable per subject, from what the texts state."""

from docta_orchestrator.realtext.questions import build
from docta_orchestrator.realtext.stated import Index


def city(q, label, years, born_here=()):
    return {"kind": "city", "qid": q, "label": label, "title": label,
            "facts": {"inception": years}, "born_here": list(born_here)}  # fmt: skip


def person(q, label, years, **links):
    return {"kind": "person", "qid": q, "label": label, "title": label,
            "facts": {"birth_year": years}, **links}  # fmt: skip


def article(q, *paragraphs):
    return {"qid": q, "title": q, "sections": [{"path": [q], "paragraphs": list(paragraphs)}]}


CITIES = [
    city("Q1", "Varna", [1749], born_here=["P1"]),
    city("Q2", "Brask", [1802]),
    city("Q3", "Tolm", [1749, 1750]),
    person("P1", "Ada Kel", [1950], born_in=["Q1"]),
]
ARTICLES = [
    article("Q1", "Varna was established in 1749. It lies on a river."),
    article("Q2", "Brask is a town on a hill."),
    article("Q3", "Tolm was founded in 1750."),
    article("P1", "Ada Kel was born in 1950 in Varna.", "In 1802 the mill burned."),
]


def by_body(questions):
    return {q["body"]: q for q in questions}


def test_a_value_counts_as_stated_in_the_own_article_or_next_to_the_name():
    index = Index(ARTICLES)
    assert index.states("Q1", "Varna", [1749])
    assert not index.states("Q2", "Brask", [1802])  # 1802 elsewhere, Brask not named
    other = article("P9", "Brask grew fast after 1802.")
    assert Index([*ARTICLES, other]).states("Q2", "Brask", [1802])


def test_retrieve_questions_use_seen_wordings_and_abstain_where_nothing_states():
    questions = [q for q in build(CITIES, ARTICLES, seed=0) if q["hops"] == 1]
    varna = next(q for q in questions if "Varna" in q["body"])
    assert varna["wording"] == "seen"
    assert any(w in varna["body"] for w in ("founded", "founding", "established"))
    assert (varna["answerable"], varna["answer"], varna["primitive"]) == (True, "1749", "retrieve")
    brask = next(q for q in questions if "Brask" in q["body"])
    assert (brask["answerable"], brask["answer"], brask["primitive"]) == (
        False, None, "spot_missing")  # fmt: skip
    assert brask["unanswerable"] == "natural" and "1802" in brask["missing"]


def test_any_stated_value_is_right_and_several_values_are_tagged():
    tolm = next(q for q in build(CITIES, ARTICLES, seed=0) if q["subjects"] == ["Q3"])
    assert tolm["values"] == ["1749", "1750"] and tolm["multi_valued"]
    assert tolm["answerable"]  # 1750 is stated


def test_a_chain_through_a_birthplace_is_a_new_wording():
    chain = next(q for q in build(CITIES, ARTICLES, seed=0) if q["hops"] == 2)
    assert "the city where Ada Kel was born" in chain["body"]
    assert (chain["wording"], chain["answer"], chain["answerable"]) == ("new", "1749", True)
    assert chain["primitive"] == "chain"


def test_compare_names_the_earlier_city_and_needs_both_years_stated():
    questions = [q for q in build(CITIES, ARTICLES, seed=0) if q["primitive"] in (
        "compare", "spot_missing") and " or " in q["body"]]  # fmt: skip
    answers = {frozenset(q["subjects"]): q for q in questions}
    assert answers[frozenset({"Q1", "Q2"})]["answerable"] is False  # Brask's year unstated
    # Varna 1749 against Tolm 1749 or 1750: overlapping values, no fair answer.
    assert frozenset({"Q1", "Q3"}) not in answers


def test_same_seed_gives_the_same_questions():
    assert build(CITIES, ARTICLES, seed=3) == build(CITIES, ARTICLES, seed=3)
    ids = [q["question_id"] for q in build(CITIES, ARTICLES, seed=3)]
    assert ids == sorted(set(ids))


def test_languages_ask_releases_and_designers_in_new_wordings():
    entities = [
        {"kind": "language", "qid": "L1", "label": "Brisk", "title": "Brisk",
         "facts": {"inception": [1995]}, "designed_by": ["P1"]},
        person("P1", "Ada Kel", [1950], designed=["L1"]),
    ]  # fmt: skip
    articles = [article("L1", "Brisk was designed by Ada Kel in 1995."),
                article("P1", "Ada Kel was born in 1950.")]  # fmt: skip
    bodies = by_body(build(entities, articles, seed=0))
    assert bodies["In what year was Brisk first released?"]["answer"] == "1995"
    chain = next(q for q in bodies.values() if "the designer of Brisk" in q["body"])
    assert (chain["answer"], chain["wording"]) == ("1950", "new")
