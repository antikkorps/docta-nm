"""Programming languages: the far transfer, relations the invented worlds
never had (designed by, influenced by), still asked as retrieve, compare or
chain."""

from docta_orchestrator.realtext.entities import add_person, entity, years
from docta_orchestrator.realtext.wikidata import batched, by_qid_order, group, qid

# Same statement rules as the cities: every non-deprecated value, no "unknown".
LANGUAGES = """SELECT ?l ?label ?article ?inc WHERE {
  ?l wdt:P31/wdt:P279* wd:Q9143; p:P571 ?si. ?si ps:P571 ?inc; wikibase:rank ?r.
  FILTER(?r != wikibase:DeprecatedRank && isLiteral(?inc))
  ?article schema:about ?l; schema:isPartOf <https://en.wikipedia.org/>.
  ?l rdfs:label ?label. FILTER(LANG(?label) = "en") }"""
DESIGNERS = """SELECT ?l ?d ?label ?article ?birth WHERE { VALUES ?l { {qids} }
  ?l wdt:P287 ?d. ?article schema:about ?d; schema:isPartOf <https://en.wikipedia.org/>.
  ?d p:P569 ?sb. ?sb ps:P569 ?birth; wikibase:rank ?r.
  FILTER(?r != wikibase:DeprecatedRank && isLiteral(?birth))
  ?d rdfs:label ?label. FILTER(LANG(?label) = "en") }"""
INFLUENCES = """SELECT ?l ?i WHERE { VALUES ?l { {qids} } ?l wdt:P737 ?i }"""


def collect(ask, limit: int) -> list[dict]:
    """The `limit` languages of lowest id that a chain can be asked about:
    a designer with an article, or an influence inside the corpus."""
    rows = group(ask(LANGUAGES), "l")
    every = by_qid_order(rows)
    designers = group(batched(ask, DESIGNERS, every), "l")
    influences = group(batched(ask, INFLUENCES, every), "l")

    def inside(q: str, corpus) -> list[str]:
        return by_qid_order(qid(r["i"]) for r in influences.get(q, []) if qid(r["i"]) in corpus)

    chosen = chainable(every, limit, lambda q, corpus: bool(designers.get(q) or inside(q, corpus)))
    found, people = [], {}
    for q in chosen:
        links = {
            "designed_by": by_qid_order(qid(r["d"]) for r in designers.get(q, [])),
            "influenced_by": inside(q, set(chosen)),
        }
        facts = {"inception": years(rows[q], "inc")}
        found.append(entity("languages", "language", q, rows[q][0], facts=facts, **links))
        for row in designers.get(q, []):
            add_person(people, "languages", row, "d", "designed", q)
    return found + [people[p] for p in by_qid_order(people)]


def chainable(every: list[str], limit: int, linked) -> list[str]:
    """The first `limit` languages still linked once the corpus is cut to
    them: an influence beyond the cut cannot carry a chain, so a language
    left with nothing is dropped and the next one takes its place."""
    pool = list(every)
    while True:
        chosen = [q for q in pool if linked(q, set(pool))][:limit]
        unlinked = [q for q in chosen if not linked(q, set(chosen))]
        if not unlinked:
            return chosen
        pool = [q for q in pool if q not in unlinked]
