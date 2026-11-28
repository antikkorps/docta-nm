"""The three domains of the delexicalisation corpus, outside the probe's
(cities, languages) so training text never shares a subject with it.

Each domain is a main query (the subject, its article, and the value every
chosen entity must have) plus values and one relation asked in batches, the
same shape three times: one collector reads all three descriptions."""

from dataclasses import dataclass

from docta_orchestrator.realtext.entities import entity, years
from docta_orchestrator.realtext.wikidata import batched, by_qid_order, group, qid

# Every non-deprecated statement, as in the probe; `isLiteral` drops "unknown".
STATED = "wikibase:rank ?r. FILTER(?r != wikibase:DeprecatedRank && isLiteral(?{v}))"
ARTICLE = """?article schema:about ?s; schema:isPartOf <https://en.wikipedia.org/>.
  ?s rdfs:label ?label. FILTER(LANG(?label) = "en")"""
# Every non-deprecated statement, not only the best-ranked (`wdt:`): an
# earlier place's name left out would stay in the text as real. `isIRI`
# drops "unknown value", which is no entity.
RELATION = """SELECT ?s ?o ?oLabel WHERE {{ VALUES ?s {{ {{qids}} }}
  ?s p:{p} ?st. ?st ps:{p} ?o; wikibase:rank ?r.
  FILTER(?r != wikibase:DeprecatedRank && isIRI(?o))
  ?o rdfs:label ?oLabel. FILTER(LANG(?oLabel) = "en") }}"""
ALIASES = """SELECT ?s ?alias WHERE { VALUES ?s { {qids} }
  ?s skos:altLabel ?alias. FILTER(LANG(?alias) = "en") }"""


def statement(prop: str, var: str, quantity: bool) -> str:
    """`?s`'s statements of `prop` bound to `var`. Quantities go through
    `psn:`, normalised to SI units: heights and lengths in metres, whatever
    unit the statement used."""
    value = f"psn:{prop}/wikibase:quantityAmount" if quantity else f"ps:{prop}"
    return f"?s p:{prop} ?st. ?st {value} ?{var}; {STATED.format(v=var)}"


def main_query(cls: str, prop: str, quantity: bool = False) -> str:
    """Instances of `cls` with a stated `prop` and an English article."""
    var = "quantity" if quantity else "year"
    return (
        f"SELECT ?s ?label ?article ?{var} WHERE {{ ?s wdt:P31 wd:{cls}. "
        f"{statement(prop, var, quantity)}\n  {ARTICLE} }}"
    )


def values_query(prop: str, quantity: bool) -> str:
    """Every stated value of `prop` for a batch of ids (`{qids}`)."""
    return (
        "SELECT ?s ?value WHERE { VALUES ?s { {qids} } " + statement(prop, "value", quantity) + " }"
    )


@dataclass(frozen=True)
class Domain:
    name: str
    kind: str
    main: str
    main_fact: str  # what the main query's value is saved as
    values: tuple[tuple[str, str], ...]  # (fact, batched query)
    relation: tuple[str, str]  # (fact, batched query)


UNIVERSITIES = Domain(
    name="universities",
    kind="university",
    main=main_query("Q3918", "P571"),
    main_fact="inception",
    values=(("students", values_query("P2196", quantity=False)),),
    relation=("located_in", RELATION.format(p="P131")),
)
MOUNTAINS = Domain(
    name="mountains",
    kind="mountain",
    main=main_query("Q8502", "P2044", quantity=True),
    main_fact="elevation_m",
    values=(("prominence_m", values_query("P2660", quantity=True)),),
    relation=("range", RELATION.format(p="P4552")),
)
BRIDGES = Domain(
    name="bridges",
    kind="bridge",
    main=main_query("Q12280", "P1619"),
    main_fact="opened",
    values=(("length_m", values_query("P2043", quantity=True)),),
    relation=("crosses", RELATION.format(p="P177")),
)
DOMAINS = {d.name: d for d in (UNIVERSITIES, MOUNTAINS, BRIDGES)}


def collect(domain: Domain, ask, limit: int) -> list[dict]:
    """The `limit` entities of lowest id, every stated value kept."""
    rows = group(ask(domain.main), "s")
    chosen = by_qid_order(rows)[:limit]
    extra = {fact: group(batched(ask, query, chosen), "s") for fact, query in domain.values}
    related = group(batched(ask, domain.relation[1], chosen), "s")
    aliases = group(batched(ask, ALIASES, chosen), "s")
    return [record(domain, q, rows[q], extra, related.get(q, []), aliases.get(q, []))
            for q in chosen]  # fmt: skip


def record(domain: Domain, q: str, rows, extra, related, aliases) -> dict:
    main = years(rows, "year") if "year" in rows[0] else numbers(rows, "quantity")
    facts = {domain.main_fact: main}
    facts |= {fact: numbers(found.get(q, []), "value") for fact, found in extra.items()}
    facts[domain.relation[0]] = linked(related)
    names = sorted({r["alias"] for r in aliases})
    return entity(domain.name, domain.kind, q, rows[0], facts=facts, aliases=names)


def linked(rows: list[dict]) -> list[dict]:
    """The related entities in id order, each with its English label."""
    ordered = sorted(rows, key=lambda r: int(qid(r["o"])[1:]))
    return [{"qid": qid(r["o"]), "label": r["oLabel"]} for r in ordered]


def numbers(rows: list[dict], field: str) -> list[float]:
    """Every stated value; whole numbers as ints, so 120 reads as in the text."""
    found = {float(r[field]) for r in rows if field in r}
    return sorted(int(v) if v.is_integer() else v for v in found)
