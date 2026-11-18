"""KeywordModel: searches with the question's words, then reads an answer next
to the subject's name with simple rules. Abstains when nothing fits.

It knows nothing of predicates: it keeps the sentence sharing the most words
with the question, then takes a number or a new proper name from it.
"""

import re

from docta_orchestrator.states import (
    Conclusion,
    Confrontation,
    Evidence,
    Exit,
    Framing,
    Verdict,
)

QUESTION_WORDS = {"In", "What", "Which", "Who", "How", "At", "The"}
NUMBER_STARTS = ("in what year", "what is the", "how ", "at what")
Found = tuple[str, tuple[str, ...]]  # (answer, cited source ids)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _names(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[A-Z][a-z]+\b", text) if w not in QUESTION_WORDS]


def _pick(sentence: str, kind: str, known: set[str]) -> str | None:
    """A number, or a proper name the question does not already contain."""
    if kind == "number":
        found = re.search(r"\b\d+\b", sentence)
        return found.group() if found else None
    fresh = [n for n in _names(sentence) if n not in known]
    return fresh[0] if fresh else None


def _best(question: str, evidence: Evidence, subject: list[str], kind: str) -> Found | None:
    """Answer from the sentence naming the subject that best overlaps the question."""
    asked, known = _words(question), set(_names(question))
    best: tuple[int, str, str] | None = None
    for hit in evidence:
        for sentence in re.split(r"(?<=\.)\s+", hit.body):
            answer = _pick(sentence, kind, known)
            if answer is None or not any(name in sentence for name in subject):
                continue
            overlap = len(asked & _words(sentence))
            if best is None or overlap > best[0]:
                best = (overlap, answer, hit.source_id)
    return (best[1], (best[2],)) if best else None


def _choice(question: str, evidence: Evidence, a: str, b: str) -> Found | None:
    found = [_best(question, evidence, _names(x), "number") for x in (a, b)]
    if None in found:
        return None
    (va, sa), (vb, sb) = found
    # "earlier" asks for the smaller value; every other comparison the larger.
    a_wins = (int(va) < int(vb)) == ("earlier" in question)
    return (a if a_wins else b), tuple(dict.fromkeys(sa + sb))


class KeywordModel:
    name = "keyword"

    def answer(self, framing: Framing, evidence: Evidence) -> Found | None:
        q = framing.question
        options = re.search(r", (.+) or (.+)\?$", q)
        if options:
            return _choice(q, evidence, options[1], options[2])
        kind = "number" if q.lower().startswith(NUMBER_STARTS) else "name"
        return _best(q, evidence, _names(q), kind)

    def frame(self, question: str) -> Framing:
        return Framing(question)

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        return [framing.question] if round_ == 0 else []

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        found = self.answer(framing, evidence) is not None
        return Verdict(True) if found else Verdict(False, Exit.NOT_FOUND)

    def confront(self, framing: Framing, evidence: Evidence) -> Confrontation:
        return Confrontation()

    def conclude(self, framing: Framing, evidence: Evidence, confrontation) -> Conclusion:
        found = self.answer(framing, evidence)
        assert found is not None, "evaluate only lets answerable evidence through"
        return Conclusion(found[0], found[1], 0.5)
