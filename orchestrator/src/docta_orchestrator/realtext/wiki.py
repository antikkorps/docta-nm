"""One Wikipedia article as plain text, pinned to the revision it came from."""

import hashlib

from docta_orchestrator.realtext.entities import SCHEMA
from docta_orchestrator.realtext.structure import sections

API = "https://en.wikipedia.org/w/api.php"
LICENCE = "CC BY-SA 4.0"


def params(title: str) -> dict:
    return {
        "action": "query",
        "prop": "extracts|revisions",
        "explaintext": "1",
        "exsectionformat": "wiki",
        "rvprop": "ids",
        "titles": title,
        "redirects": "1",
        "format": "json",
        "formatversion": "2",
    }


def article_record(entity: dict, answer: dict) -> dict | None:
    """The article as sources to read: structure and provenance, no facts.
    The oldid link reaches this exact text even after the page changes."""
    page = answer["query"]["pages"][0]
    if page.get("missing") or "extract" not in page:
        return None
    revid = page["revisions"][0]["revid"]
    return {
        "schema": SCHEMA,
        "type": "article",
        "domain": entity["domain"],
        "qid": entity["qid"],
        "title": page["title"],
        "url": f"https://en.wikipedia.org/w/index.php?oldid={revid}",
        "licence": LICENCE,
        "sha256": hashlib.sha256(page["extract"].encode()).hexdigest(),
        "sections": sections(page["title"], page["extract"]),
    }
