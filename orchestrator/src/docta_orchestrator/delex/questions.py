"""Questions on the delexicalised corpus, from its rewritten answer key.

Each domain asks its facts as a year, a quantity or a name, in the worlds'
own wordings where a relation has one (founded, based in, elevation), so a
gain on them shows reading, not parsing; the others get new wordings and
say so. Only retrieve and compare: a chain's second hop (the university's
city and that city's founding) has no article in this corpus. Whether a
question is answerable is judged on the texts: the subject's own article
must state the rewritten value, which the rewriting put there itself."""

import random
from itertools import combinations

from docta_orchestrator.models.templates import COMPARISONS, QUESTIONS
from docta_orchestrator.realtext.entities import SCHEMA
from docta_orchestrator.realtext.questions import named, question
from docta_orchestrator.realtext.stated import MIN_NAME, Index, mentions

# kind → (fact, value kind, seen predicate or None, new wording or None)
FACTS = {
    "university": [("inception", "year", "founded_in", None),
                   ("located_in", "name", "based_in", None),
                   ("students", "quantity", None, "How many students does {s} have?")],
    "mountain": [("elevation_m", "quantity", "elevation", None),
                 ("range", "name", None, "In which mountain range does {s} lie?")],
    "bridge": [("opened", "year", None, "In what year did {s} open?"),
               ("crosses", "name", None, "What does {s} cross?"),
               ("length_m", "quantity", None, "How long is {s}, in metres?")],
}  # fmt: skip
# kind → (fact, wording, seen, whether the larger value wins)
COMPARE = {
    "university": ("inception", COMPARISONS["founded_in"], True, False),
    "mountain": ("elevation_m", COMPARISONS["elevation"], True, True),
    "bridge": ("opened", "Which opened earlier, {s} or {o}?", False, False),
}
KINDS = {f: k for specs in FACTS.values() for f, k, _, _ in specs}

__all__ = ["Index", "build", "stated", "stating"]


def forms(fact: str, value) -> list[str]:
    """How the text writes a value: a year as is, a quantity with or without
    thousands commas, a related entity by its label (too short a label
    matches initials, so it never counts)."""
    if KINDS[fact] == "name":
        return [value["label"]] if len(value["label"]) >= MIN_NAME else []
    if KINDS[fact] == "quantity":
        return list(dict.fromkeys([f"{value:,}", str(value)]))
    return [str(value)]


def stating(index: Index, entity: dict, fact: str) -> list[tuple[str, str, str]]:
    """Every (article, sentence, form) of the subject's article stating one
    of the fact's values."""
    own = index.own.get(entity["qid"], [])
    return [(entity["qid"], s, f) for v in entity["facts"].get(fact, [])
            for f in forms(fact, v) for s in own if mentions(s, f)]  # fmt: skip


def stated(index: Index, entity: dict, fact: str) -> str | None:
    """The value as the text writes it, if the subject's article states one."""
    found = stating(index, entity, fact)
    return found[0][2] if found else None


def keyed(entity: dict, fact: str) -> list[str]:
    return [v["label"] if isinstance(v, dict) else str(v) for v in entity["facts"].get(fact, [])]


def retrieve(e: dict, spec: tuple, index: Index, rng: random.Random) -> dict | None:
    fact, _, seen, new = spec
    if not e["facts"].get(fact):
        return None
    body = rng.choice(QUESTIONS[seen]) if seen else new
    gap = f"no text states the {fact.replace('_', ' ')} of {e['name']}"
    q = question(body.format(s=e["name"]), 1, "retrieve", stated(index, e, fact),
                 keyed(e, fact), [e["qid"]], bool(seen), gap)  # fmt: skip
    return q | {"fact": fact}


def apart(a: dict, b: dict, fact: str) -> bool:
    va, vb = a["facts"].get(fact), b["facts"].get(fact)
    return bool(va and vb) and (max(va) < min(vb) or max(vb) < min(va))


def compare(group: list[dict], index: Index, rng: random.Random) -> list[dict]:
    """As many pairs as entities, among those whose values cannot overlap
    whatever stated value is right."""
    if not group:
        return []
    fact, wording, seen, larger = COMPARE[group[0]["kind"]]
    pairs = [(a, b) for a, b in combinations(group, 2) if apart(a, b, fact)]
    out = []
    for a, b in rng.sample(pairs, min(len(group), len(pairs))):
        low_first = max(a["facts"][fact]) < min(b["facts"][fact])
        winner = (b if low_first else a) if larger else (a if low_first else b)
        both = all(stated(index, x, fact) for x in (a, b))
        gap = f"no text states the {fact.replace('_', ' ')} of both {a['name']} and {b['name']}"
        q = question(wording.format(s=a["name"], o=b["name"]), 1, "compare",
                     winner["name"] if both else None, [winner["name"]],
                     [a["qid"], b["qid"]], seen, gap)  # fmt: skip
        out.append(q | {"fact": fact})
    return out


def build(entities: list[dict], articles: list[dict], seed: int = 0) -> list[dict]:
    """Every retrieve question the key allows, then up to one comparison per
    entity of each kind, numbered in a fixed order."""
    rng, index = random.Random(seed), Index(articles)
    entities = named(entities)
    out = [q for e in entities for spec in FACTS.get(e["kind"], [])
           if (q := retrieve(e, spec, index, rng))]  # fmt: skip
    for kind in FACTS:
        out += compare([e for e in entities if e["kind"] == kind], index, rng)
    return [{"schema": SCHEMA, "type": "question", "question_id": f"q{i:05d}", **q}
            for i, q in enumerate(out)]  # fmt: skip
