"""Each domain picks its entities in a fixed order and keeps every value
Wikidata states, so disagreements between statements survive."""

from docta_orchestrator.realtext import cities, languages

E = "http://www.wikidata.org/entity/"
W = "https://en.wikipedia.org/wiki/"


def row(subject, key, q, label=None, **values):
    """A query row: the subject's id under `key`, plus a named entity if given."""
    found = {key: E + subject, **values}
    if label:
        found |= {"label": label, "article": W + label}
    if q:
        found[q[0]] = E + q[1]
    return found


def fake(answers):
    """Answers each query by the first marker it contains."""

    def ask(query):
        return next(rows for marker, rows in answers.items() if marker in query)

    return ask


CITY_ROWS = [
    row("Q20", "c", None, "Brask", inc="1820-01-01T00:00:00Z", elev="210"),
    row("Q20", "c", None, "Brask", inc="1821-01-01T00:00:00Z", elev="210"),
    row("Q9", "c", None, "Varnelle", inc="1700-01-01T00:00:00Z", elev="5"),
    row("Q100", "c", None, "Orlo", inc="1900-01-01T00:00:00Z", elev="1"),
]
POPULATION = [
    {"c": E + "Q20", "pop": "12000", "when": "2010-01-01T00:00:00Z"},
    {"c": E + "Q20", "pop": "12500", "when": "2020-01-01T00:00:00Z"},
    {"c": E + "Q9", "pop": "900"},
]
BORN = [
    row("Q20", "c", ("p", "Q7"), "Ana", birth="1950-01-01T00:00:00Z"),
    row("Q20", "c", ("p", "Q3"), "Bo", birth="1960-01-01T00:00:00Z"),
    row("Q20", "c", ("p", "Q5"), "Cy", birth="1970-01-01T00:00:00Z"),
]
ASK_CITIES = fake({"P2044": CITY_ROWS, "p:P1082": POPULATION, "P19": BORN})


def test_cities_are_the_lowest_ids_and_keep_every_founding_year():
    found = [e for e in cities.collect(ASK_CITIES, limit=2) if e["kind"] == "city"]
    assert [e["qid"] for e in found] == ["Q9", "Q20"]
    assert found[1]["facts"]["inception"] == [1820, 1821]
    assert found[1]["title"] == "Brask"


def test_population_keeps_each_statement_with_its_year():
    brask = next(e for e in cities.collect(ASK_CITIES, limit=2) if e["qid"] == "Q20")
    assert brask["facts"]["population"] == [
        {"value": 12000, "year": 2010},
        {"value": 12500, "year": 2020},
    ]


def test_two_people_born_in_each_city_link_it():
    found = cities.collect(ASK_CITIES, limit=2)
    people = [e for e in found if e["kind"] == "person"]
    assert [p["qid"] for p in people] == ["Q3", "Q5"]
    assert people[0]["born_in"] == ["Q20"] and people[0]["facts"]["birth_year"] == [1960]
    assert next(e for e in found if e["qid"] == "Q20")["born_here"] == ["Q3", "Q5"]


LANG_ROWS = [
    row("Q30", "l", None, "Kel", inc="1990-01-01T00:00:00Z"),
    row("Q31", "l", None, "Mor", inc="1995-01-01T00:00:00Z"),
    row("Q32", "l", None, "Lone", inc="2000-01-01T00:00:00Z"),
]
DESIGNERS = [row("Q30", "l", ("d", "Q8"), "Dee", birth="1940-01-01T00:00:00Z")]
INFLUENCES = [{"l": E + "Q31", "i": E + "Q30"}, {"l": E + "Q32", "i": E + "Q999"}]
ASK_LANGUAGES = fake({"Q9143": LANG_ROWS, "P287": DESIGNERS, "P737": INFLUENCES})


def test_languages_need_a_designer_or_an_influence_inside_the_corpus():
    # Lone's only influence is outside the corpus: no chain can be asked about it.
    found = [e for e in languages.collect(ASK_LANGUAGES, limit=5) if e["kind"] == "language"]
    assert [e["qid"] for e in found] == ["Q30", "Q31"]
    assert found[1]["influenced_by"] == ["Q30"]


def test_designers_come_with_their_birth_year():
    found = languages.collect(ASK_LANGUAGES, limit=5)
    dee = next(e for e in found if e["qid"] == "Q8")
    assert dee["kind"] == "person" and dee["facts"]["birth_year"] == [1940]
    assert next(e for e in found if e["qid"] == "Q30")["designed_by"] == ["Q8"]


def test_a_language_whose_only_influence_falls_beyond_the_cut_gives_its_place():
    # Q2 is linked only through Q4; cut to two, Q4 is out, so Q2 makes way for Q3.
    links = {"Q1": {"designer"}, "Q2": {"Q4"}, "Q3": {"designer"}, "Q4": {"designer"}}

    def linked(q, corpus):
        return "designer" in links[q] or bool(links[q] & corpus)

    assert languages.chainable(["Q1", "Q2", "Q3", "Q4"], 2, linked) == ["Q1", "Q3"]
