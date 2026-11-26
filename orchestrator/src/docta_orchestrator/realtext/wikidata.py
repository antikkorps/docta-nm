"""Wikidata facts are the probe's answer key: used to score, never read by a
model, like the facts of an invented world."""

from urllib.parse import unquote

ENTITY = "http://www.wikidata.org/entity/"
WIKI = "https://en.wikipedia.org/wiki/"


def bindings(payload: dict) -> list[dict[str, str]]:
    return [{k: v["value"] for k, v in b.items()} for b in payload["results"]["bindings"]]


def qid(uri: str) -> str:
    return uri.removeprefix(ENTITY)


def title_of(article_url: str) -> str:
    return unquote(article_url.removeprefix(WIKI)).replace("_", " ")


def year(value: str) -> int:
    """'-0386-01-01T00:00:00Z' -> -386: founding dates go back before our era."""
    sign = -1 if value.startswith("-") else 1
    return sign * int(value.lstrip("-+").split("-")[0])


def by_qid_order(qids) -> list[str]:
    """A fixed order that needs no ranking: the same selection on every run."""
    return sorted(set(qids), key=lambda q: int(q[1:]))


def group(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(qid(row[key]), []).append(row)
    return grouped


def batched(ask, query: str, qids: list[str], size: int = 50) -> list[dict]:
    """One query per batch of ids: short enough for a GET, few enough requests."""
    rows: list[dict] = []
    for start in range(0, len(qids), size):
        chunk = " ".join(f"wd:{q}" for q in qids[start : start + size])
        rows += ask(query.replace("{qids}", chunk))
    return rows
