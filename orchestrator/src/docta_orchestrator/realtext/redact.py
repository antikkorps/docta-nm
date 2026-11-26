"""Redaction: a controlled share of facts made unanswerable by cutting, from
what the core may read, every sentence that states them.

Natural absence alone leaves a fifth of the questions unanswerable, and only
where the texts happen to be silent. Cutting gives the third the worlds use,
on every shape. The cuts are records (redactions.jsonl) applied at load
time; articles.jsonl stays the text as collected.
"""

import random

from docta_orchestrator.realtext.entities import SCHEMA
from docta_orchestrator.realtext.questions import asked, named
from docta_orchestrator.realtext.stated import Index, sentences


def redact(entities: list[dict], articles: list[dict], share: float, seed: int) -> list[dict]:
    """Cuts for enough stated facts that `share` of the asked facts have no
    answer, counting those the texts already leave out."""
    index = Index(articles)
    facts = [e for e in named(entities) if asked(e)]
    stated = [e for e in facts if index.states(e["qid"], e["name"], asked(e)[1])]
    need = round(share * len(facts)) - (len(facts) - len(stated))
    chosen = random.Random(seed).sample(stated, max(0, min(need, len(stated))))
    out = []
    for e in sorted(chosen, key=lambda e: e["qid"]):
        cut = dict.fromkeys(index.stating(e["qid"], e["name"], asked(e)[1]))
        out += [{"schema": SCHEMA, "type": "redaction", "subject": e["qid"],
                 "fact": asked(e)[0], "article": q, "sentence": s} for q, s in cut]  # fmt: skip
    return out


def apply(articles: list[dict], cuts: list[dict]) -> list[dict]:
    """The articles without the cut sentences, as new records; a paragraph
    left empty is dropped."""
    gone: dict[str, set[str]] = {}
    for c in cuts:
        gone.setdefault(c["article"], set()).add(c["sentence"])
    return [{**a, "sections": [
        {**s, "paragraphs": kept(s["paragraphs"], gone.get(a["qid"], set()))}
        for s in a["sections"]]} for a in articles]  # fmt: skip


def kept(texts: list[str], gone: set[str]) -> list[str]:
    out = [" ".join(s for s in sentences(p) if s not in gone) for p in texts]
    return [p for p in out if p]


def tag(original: list[dict], redacted: list[dict]) -> list[dict]:
    """Each question on the redacted texts, marked `redacted` where only the
    cuts took its answer away. Both lists come from the same seed, so they
    hold the same questions in the same order."""
    out = []
    for before, after in zip(original, redacted, strict=True):
        assert before["body"] == after["body"], "built from different seeds"
        made = before["answerable"] and not after["answerable"]
        out.append({**after, "unanswerable": "redacted"} if made else after)
    return out
