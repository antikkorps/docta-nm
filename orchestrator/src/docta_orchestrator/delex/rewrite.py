"""One article and its entity, rewritten with invented names and values.

Names map word by word, so a partial mention ("Everest" after "Mount
Everest") is replaced like the full one; the facts go through the same
names and the same shift and factor, so a label still matches the text."""

import random

from docta_orchestrator.delex import numbers
from docta_orchestrator.delex.words import WORD

# Shifts only back in time, so no date lands in the future; a factor kept
# away from 1, so no value stays nearly real.
SHIFT = (30, 150)
FACTORS = ((0.6, 0.9), (1.1, 1.6))
# Facts that are dates; every other number is a quantity.
YEAR_FACTS = {"inception", "opened"}


def draw(rng: random.Random) -> tuple[int, float]:
    shift = -rng.randint(*SHIFT)
    low, high = FACTORS[rng.randrange(2)]
    return shift, round(rng.uniform(low, high), 3)


def names_in(text: str, names: dict[str, str]) -> str:
    return WORD.sub(lambda m: names.get(m.group(0), m.group(0)), text)


def text_of(text: str, names: dict[str, str], shift: int, factor: float) -> str:
    return numbers.rewrite(names_in(text, names), shift, factor)


def article(record: dict, names: dict, shift: int, factor: float) -> dict:
    sections = [{"path": [names_in(p, names) for p in s["path"]],
                 "paragraphs": [text_of(p, names, shift, factor) for p in s["paragraphs"]]}
                for s in record["sections"]]  # fmt: skip
    # The provenance (oldid url, hash of the real text) points back at the
    # real article: the rewritten record drops both, the raw corpus keeps them.
    out = {k: v for k, v in record.items() if k not in ("url", "sha256")}
    return out | {"title": names_in(record["title"], names), "sections": sections,
                  "links": [names_in(t, names) for t in record.get("links", [])]}  # fmt: skip


def fact(value, names: dict, shift: int, factor: float, is_year: bool):
    if isinstance(value, dict):
        return {**value, "label": names_in(value["label"], names)}
    return numbers.year(value, shift) if is_year else numbers.quantity(value, factor)


def entity(record: dict, names: dict, shift: int, factor: float) -> dict:
    facts = {k: [fact(v, names, shift, factor, k in YEAR_FACTS) for v in vs]
             for k, vs in record["facts"].items()}  # fmt: skip
    return record | {"label": names_in(record["label"], names),
                     "title": names_in(record["title"], names),
                     "aliases": [names_in(a, names) for a in record.get("aliases", [])],
                     "facts": facts}  # fmt: skip


def pair(art: dict, ent: dict, names: dict, rng: random.Random) -> tuple[dict, dict]:
    shift, factor = draw(rng)
    return article(art, names, shift, factor), entity(ent, names, shift, factor)
