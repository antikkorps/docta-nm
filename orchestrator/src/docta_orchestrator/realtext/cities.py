"""Cities of the United States: the near transfer, with the relations of the
invented worlds (founded, population, elevation, who comes from there)."""

from docta_orchestrator.realtext.entities import add_person, entity, years
from docta_orchestrator.realtext.wikidata import batched, by_qid_order, group, qid, year

# Every statement but deprecated ones (`p:`/`ps:`), not only the best-ranked
# (`wdt:`), so disagreements survive; `isLiteral` drops "unknown value".
# Elevation goes through `psn:`, normalised to metres: many are stated in feet.
# Mid-sized cities: articles a person can check, births few enough to query.
CITIES = """SELECT ?c ?label ?article ?inc ?elev WHERE {
  ?c wdt:P31/wdt:P279* wd:Q515; wdt:P17 wd:Q30.
  FILTER EXISTS { ?c wdt:P1082 ?pop. FILTER(?pop >= 10000 && ?pop <= 300000) }
  ?c p:P571 ?si. ?si ps:P571 ?inc; wikibase:rank ?ri. FILTER(?ri != wikibase:DeprecatedRank)
  ?c p:P2044 ?se. ?se psn:P2044/wikibase:quantityAmount ?elev; wikibase:rank ?re.
  FILTER(?re != wikibase:DeprecatedRank && isLiteral(?inc))
  ?article schema:about ?c; schema:isPartOf <https://en.wikipedia.org/>.
  ?c rdfs:label ?label. FILTER(LANG(?label) = "en") }"""
POPULATION = """SELECT ?c ?pop ?when WHERE { VALUES ?c { {qids} }
  ?c p:P1082 ?st. ?st ps:P1082 ?pop; wikibase:rank ?r.
  FILTER(?r != wikibase:DeprecatedRank && isLiteral(?pop))
  OPTIONAL { ?st pq:P585 ?when. FILTER(isLiteral(?when)) } }"""
BORN = """SELECT ?c ?p ?label ?article ?birth WHERE { VALUES ?c { {qids} }
  ?p wdt:P19 ?c; p:P569 ?sb. ?sb ps:P569 ?birth; wikibase:rank ?r.
  FILTER(?r != wikibase:DeprecatedRank && isLiteral(?birth))
  ?article schema:about ?p; schema:isPartOf <https://en.wikipedia.org/>.
  ?p rdfs:label ?label. FILTER(LANG(?label) = "en") }"""
PEOPLE_PER_CITY = 2


def collect(ask, limit: int) -> list[dict]:
    """The `limit` cities of lowest id, then two people born in each."""
    rows = group(ask(CITIES), "c")
    chosen = by_qid_order(rows)[:limit]
    pops = group(batched(ask, POPULATION, chosen), "c")
    # Births are the heaviest query: small batches stay under the 60 s limit.
    born = group(batched(ask, BORN, chosen, size=5), "c")
    found, people = [], {}
    for q in chosen:
        picked = by_qid_order(qid(r["p"]) for r in born.get(q, []))[:PEOPLE_PER_CITY]
        found.append(city(q, rows[q], pops.get(q, []), picked))
        for row in (r for r in born.get(q, []) if qid(r["p"]) in picked):
            add_person(people, "cities", row, "p", "born_in", q)
    return found + [people[p] for p in by_qid_order(people)]


def city(q: str, rows: list[dict], pops: list[dict], born_here: list[str]) -> dict:
    facts = {
        "inception": years(rows, "inc"),
        "elevation_m": sorted({float(r["elev"]) for r in rows}),
        "population": population(pops),
    }
    return entity("cities", "city", q, rows[0], facts=facts, born_here=born_here)


def population(rows: list[dict]) -> list[dict]:
    """Every census figure with its year: a city has several, and sources
    quoting different years disagree without either being wrong."""
    seen = {(round(float(r["pop"])), year(r["when"]) if "when" in r else None) for r in rows}
    ordered = sorted(seen, key=lambda s: (s[1] is None, s[1] or 0, s[0]))
    return [{"value": v, "year": y} for v, y in ordered]
