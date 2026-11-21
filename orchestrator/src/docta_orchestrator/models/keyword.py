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

# Capitalised only because they open a question or a sentence: never names.
QUESTION_WORDS = {"In", "What", "Which", "Who", "How", "At", "The", "On"}
NAME = r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b"
Found = tuple[str, tuple[str, ...]]  # (answer, cited source ids)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def capitalised_words(text: str) -> list[str]:
    return [w for w in re.findall(r"\b[A-Z][a-z]+\b", text) if w not in QUESTION_WORDS]


def asked_kind(question: str) -> str:
    """Only "which" and "who" ask for a name; every other question a number."""
    return "name" if re.search(r"\b(?:which|who)\b", question.lower()) else "number"


def _pick(sentence: str, kind: str, known: set[str]) -> str | None:
    """A number, or a proper name the question does not already contain."""
    if kind == "number":
        found = re.search(r"\b\d+\b", sentence)
        return found.group() if found else None
    fresh = [n for n in capitalised_words(sentence) if n not in known]
    return fresh[0] if fresh else None


def scored_answer(
    question: str, evidence: Evidence, subject: list[str], kind: str
) -> tuple[int, str, str] | None:
    """(overlap, answer, source id) from the sentence naming the subject that
    best overlaps the question: the overlap says how well the answer fits."""
    asked, known = _words(question), set(capitalised_words(question))
    best: tuple[int, str, str] | None = None
    for hit in evidence:
        for sentence in re.split(r"(?<=\.)\s+", hit.body):
            answer = _pick(sentence, kind, known)
            if answer is None or not any(name in sentence for name in subject):
                continue
            overlap = len(asked & _words(sentence))
            if best is None or overlap > best[0]:
                best = (overlap, answer, hit.source_id)
    return best


def best_sentence(question: str, evidence: Evidence, subject: list[str], kind: str) -> Found | None:
    best = scored_answer(question, evidence, subject, kind)
    return (best[1], (best[2],)) if best else None


def _choice(question: str, evidence: Evidence, a: str, b: str) -> Found | None:
    found = [best_sentence(question, evidence, capitalised_words(x), "number") for x in (a, b)]
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
        # Rules on what a question asks, never on its wording: the ood split
        # words questions differently, and a rule fitted to train's openings
        # would hide whatever the judgement gets right there.
        # The opening word is skipped: capitalised, it would join the first name.
        opening = re.match(r"\W*\w+", q)
        options = re.search(rf"({NAME}) or ({NAME})", q[opening.end() if opening else 0 :])
        if options:
            return _choice(q, evidence, options[1], options[2])
        return best_sentence(q, evidence, capitalised_words(q), asked_kind(q))

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
