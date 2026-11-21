"""DigModel: keyword, plus a second search driven by what the first one revealed.

"In what year was the city where Orlan Teste lives founded?": the first search
reads "Orlan Teste lives in Varnelle." Varnelle is the unknown that reading
revealed, so the next search goes after Varnelle, and the answer is read there.

It only digs when the question refers to something indirectly ("the city
where", "the guild that"). A direct question it cannot answer stays
unanswered: following whatever names come up would turn a missing fact into
a confident wrong answer. That cue is the generator's chain wording, which
every split shares, ood included; sentences and questions are read with no
template.
"""

import re

from docta_orchestrator.models.keyword import (
    NUMBER_STARTS,
    QUESTION_WORDS,
    Found,
    KeywordModel,
    capitalised_words,
    scored_answer,
)
from docta_orchestrator.states import Evidence, Exit, Framing, Verdict

Link = tuple[str, str]  # (revealed name, id of the source whose sentence revealed it)


def indirect(question: str) -> bool:
    return re.search(r"\bthe \w+ (?:where|that)\b", question) is not None


def proper_names(sentence: str) -> list[str]:
    names = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", sentence)
    # "In Varnelle, ..." opens with a capitalised preposition glued to the name.
    trimmed = [" ".join(w for w in n.split() if w not in QUESTION_WORDS) for n in names]
    return [n for n in trimmed if n]


def links(question: str, evidence: Evidence) -> list[Link]:
    """Each name a sentence about the question's subject brings in."""
    subject = set(capitalised_words(question))
    found = []
    for hit in evidence:
        for sentence in re.split(r"(?<=\.)\s+", hit.body):
            if subject & set(capitalised_words(sentence)):
                found += [(n, hit.source_id) for n in proper_names(sentence)]
    return list(dict.fromkeys((n, s) for n, s in found if not subject & set(n.split())))


def revealed(question: str, evidence: Evidence) -> list[str]:
    return list(dict.fromkeys(n for n, _ in links(question, evidence)))


def can_dig(question: str, evidence: Evidence) -> bool:
    return indirect(question) and bool(revealed(question, evidence))


def dug_answer(question: str, evidence: Evidence, only: set[str] | None = None) -> Found | None:
    """The answer read about a revealed name, citing the link that led there.
    A subject has several links (a city, a guild): the answer whose sentence
    best fits the question wins, not the first link met. `only` keeps the
    names a judge accepted as the bridge, so the answer comes from one of them."""
    kind = "number" if question.lower().startswith(NUMBER_STARTS) else "name"
    found = [(n, s) for n, s in links(question, evidence) if only is None or n in only]
    best: tuple[int, str, tuple[str, ...]] | None = None
    for name in dict.fromkeys(n for n, _ in found):
        # The revealed name joins the question, so it is never its own answer;
        # it is matched whole ("Kel Dorn", not any "Kel").
        scored = scored_answer(f"{question} {name}", evidence, [name], kind)
        if scored and (best is None or scored[0] > best[0]):
            via = tuple(s for n, s in found if n == name)
            best = (scored[0], scored[1], tuple(dict.fromkeys((*via, scored[2]))))
    return (best[1], best[2]) if best else None


class DigModel(KeywordModel):
    name = "dig"

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        q = framing.question
        if round_ == 0:
            return [q]
        if round_ > 1 or not can_dig(q, evidence):
            return []
        # Drop the subject's words: its sources were read already and would
        # crowd the new name's out of the top results.
        rest = " ".join(w for w in re.findall(r"\w+", q) if w not in capitalised_words(q))
        return [f"{name} {rest}" for name in revealed(q, evidence)]

    def answer(self, framing: Framing, evidence: Evidence) -> Found | None:
        q = framing.question
        return dug_answer(q, evidence) if indirect(q) else super().answer(framing, evidence)

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        if self.answer(framing, evidence) is not None:
            return Verdict(True)
        if can_dig(framing.question, evidence):
            return Verdict(False)  # not enough yet: back to Plan, to dig
        return Verdict(False, Exit.NOT_FOUND)
