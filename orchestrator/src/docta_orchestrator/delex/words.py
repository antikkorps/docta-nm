"""Which words are names, and what replaces them.

No trained model may find the names, so a word is a name when the sources
say so: it belongs to a subject, an alias, a linked page or a related
entity somewhere in the domain's corpus, and it is never written in
lowercase anywhere in that corpus (a word seen lowercase, like "range", is
a common noun even when a title capitalises it)."""

import re
from collections.abc import Iterable

# A capitalised run of letters, any script: "Köprüsü" is a name too.
WORD = re.compile(r"[^\W\d_]+")


def capitalised(text: str) -> set[str]:
    return {w for w in WORD.findall(text) if len(w) > 1 and w[0].isupper()}


def texts(article: dict) -> Iterable[str]:
    for section in article["sections"]:
        yield from section["path"]
        yield from section["paragraphs"]


def named(entity: dict) -> Iterable[str]:
    """Every string the answer key names: label, title, aliases, related."""
    yield entity["label"]
    yield entity["title"]
    yield from entity.get("aliases", [])
    for values in entity["facts"].values():
        yield from (v["label"] for v in values if isinstance(v, dict) and "label" in v)


def lowercase_words(articles: list[dict]) -> set[str]:
    """Every word written in lowercase somewhere: common words of the corpus."""
    return {w for a in articles for t in texts(a) for w in WORD.findall(t) if w.islower()}


def entity_words(articles: list[dict], entities: list[dict]) -> set[str]:
    """The capitalised words of every name the corpus links or the key names,
    minus those the corpus also writes in lowercase."""
    sources = [t for e in entities for t in named(e)]
    sources += [t for a in articles for t in a.get("links", [])]
    found = set().union(*(capitalised(t) for t in sources))
    common = lowercase_words(articles)
    return {w for w in found if w.lower() not in common}


class Namer:
    """Hands out invented names from a pool, skipping any the corpus already
    writes as a word, so a replacement never reads as an existing word."""

    def __init__(self, pool: list[str], forbidden: set[str]):
        self.pool, self.forbidden = pool, forbidden

    def assign(self, originals: set[str]) -> dict[str, str]:
        """Sorted originals take pool names in order: the same corpus maps the
        same way whatever order its articles came in."""
        free = (n for n in self.pool if n.lower() not in self.forbidden)
        out = {}
        for original in sorted(originals):
            name = next(free, None)
            if name is None:
                raise ValueError(f"name pool too small for {len(originals)} names")
            out[original] = name
        return out
