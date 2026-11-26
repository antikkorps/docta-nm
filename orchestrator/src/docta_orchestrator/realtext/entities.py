"""Answer-key records shared by every domain: one per entity, all of its
stated values kept, since sources and statements may disagree."""

from docta_orchestrator.realtext.wikidata import qid, title_of, year

SCHEMA = "real-v0"


def entity(domain: str, kind: str, q: str, row: dict, **extra) -> dict:
    return {
        "schema": SCHEMA,
        "type": "entity",
        "domain": domain,
        "kind": kind,
        "qid": q,
        "label": row["label"],
        "title": title_of(row["article"]),
        **extra,
    }


def years(rows: list[dict], field: str) -> list[int]:
    return sorted({year(r[field]) for r in rows if field in r})


def add_person(people: dict, domain: str, row: dict, key: str, link: str, target: str) -> None:
    """The person at `row[key]`, linked to `target` under `link` (born_in,
    designed): the first hop of a chain question."""
    q = qid(row[key])
    person = people.setdefault(
        q, entity(domain, "person", q, row, facts={"birth_year": []}, **{link: []})
    )
    person["facts"]["birth_year"] = sorted(
        set(person["facts"]["birth_year"]) | set(years([row], "birth"))
    )
    if target not in person[link]:
        person[link].append(target)
