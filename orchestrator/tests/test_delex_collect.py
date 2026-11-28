"""The delexicalisation corpus: three domains outside the probe, each entity
with every stated value, its aliases, and its article's internal links (the
spans the delexicaliser will replace)."""

from docta_orchestrator.delex import domains, links

E = "http://www.wikidata.org/entity/"
W = "https://en.wikipedia.org/wiki/"


def subject(q, label, **values):
    return {"s": E + q, "label": label, "article": W + label.replace(" ", "_"), **values}


def fake(answers):
    """Answers each query by the first marker it contains."""

    def ask(query):
        return next(rows for marker, rows in answers.items() if marker in query)

    return ask


BRIDGES = fake({
    "wd:Q12280": [
        subject("Q30", "Brask Bridge", year="1901-01-01T00:00:00Z"),
        subject("Q30", "Brask Bridge", year="1902-01-01T00:00:00Z"),
        subject("Q7", "Orlo Bridge", year="1850-01-01T00:00:00Z"),
        subject("Q900", "Late Bridge", year="1990-01-01T00:00:00Z"),
    ],
    "P2043": [{"s": E + "Q30", "value": "120.5"}, {"s": E + "Q7", "value": "40"}],
    "P177": [{"s": E + "Q30", "o": E + "Q5", "oLabel": "Varn River"}],
    "altLabel": [{"s": E + "Q30", "alias": "Old Brask"}],
})  # fmt: skip


def test_a_domain_keeps_the_lowest_ids_and_every_stated_year():
    found = domains.collect(domains.BRIDGES, BRIDGES, limit=2)
    assert [e["qid"] for e in found] == ["Q7", "Q30"]
    assert found[1]["facts"]["opened"] == [1901, 1902]
    assert found[1]["kind"] == "bridge" and found[1]["domain"] == "bridges"


def test_numbers_relations_and_aliases_come_with_the_entity():
    brask = domains.collect(domains.BRIDGES, BRIDGES, limit=2)[1]
    assert brask["facts"]["length_m"] == [120.5]
    assert brask["facts"]["crosses"] == [{"qid": "Q5", "label": "Varn River"}]
    assert brask["aliases"] == ["Old Brask"]
    assert domains.collect(domains.BRIDGES, BRIDGES, limit=2)[0]["facts"]["crosses"] == []


def test_the_three_domains_stay_outside_the_probe():
    assert set(domains.DOMAINS) == {"universities", "mountains", "bridges"}


class Pages:
    """A links API answering in two pages, as Wikipedia does past 500 links."""

    def __init__(self):
        self.asked = []

    def json(self, url, params):
        self.asked.append(params)
        if "plcontinue" not in params:
            page = {"links": [{"title": "Varn River"}, {"title": "Orlo"}]}
            return {"query": {"pages": [page]}, "continue": {"plcontinue": "x|1"}}
        return {"query": {"pages": [{"links": [{"title": "Brask (town)"}]}]}}


def test_links_follow_every_page_of_the_answer():
    fetch = Pages()
    assert links.of(fetch, "Brask Bridge") == ["Varn River", "Orlo", "Brask (town)"]
    assert fetch.asked[1]["plcontinue"] == "x|1"
    assert all(p["plnamespace"] == "0" for p in fetch.asked)


class Wiki:
    """Wikipedia for two titles: an extract for one, nothing for the other."""

    def json(self, url, params):
        if params.get("prop") == "links":
            return {"query": {"pages": [{"links": [{"title": "Varn River"}]}]}}
        if params["titles"] == "Gone Bridge":
            return {"query": {"pages": [{"missing": True}]}}
        page = {"title": params["titles"], "extract": "Brask Bridge crosses the Varn River.",
                "revisions": [{"revid": 42}]}  # fmt: skip
        return {"query": {"pages": [page]}}


def test_an_article_carries_its_links_and_one_without_text_is_skipped():
    from docta_orchestrator.delex import collect

    found = [{"qid": "Q30", "domain": "bridges", "title": "Brask Bridge"},
             {"qid": "Q31", "domain": "bridges", "title": "Gone Bridge"}]  # fmt: skip
    records = list(collect.articles(Wiki(), found))
    assert [r["qid"] for r in records] == ["Q30"]
    assert records[0]["links"] == ["Varn River"]
    assert records[0]["url"].endswith("oldid=42")
