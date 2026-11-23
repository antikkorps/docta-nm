"""CoreBridgeModel: core_read with no rule on a chain's wording.

"In what year was the city Orlan Teste resides in founded?" is read whole.
The Read core has two pointers on one reading: the answer, and the name to
search next (Varnelle, from "Orlan Teste resides in Varnelle."). Code
searches that name, as dig does, and the core reads again. The Evaluate
core judges the whole question on all that was read. No regex recognises
a chain, so a held-out link wording is no wall: whether the core finds the
bridge is what ood measures.

A human hint reaches Plan only: the core may point at a name the hint says,
which steers the search, but the answer and its citations stand on sources.
"""

import re

from docta_orchestrator.models.dig import links
from docta_orchestrator.models.keyword import Found, capitalised_words, compared
from docta_orchestrator.models.templates import TEMPLATE_WORDS
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Evidence, Exit, Framing, Verdict

from docta_train.read_model import CoreReadModel, stated


class CoreBridgeModel(CoreReadModel):
    name = "core_bridge"

    def answer(self, framing: Framing, evidence: Evidence) -> Found | None:
        q, read = framing.question, list(evidence)
        options = compared(q)
        if options:
            return self.compare(q, options, read)
        found = self.reader.read(q, read)
        if not found or not stated(found[0]):
            return None
        via = self.via(q, read, found)
        if via is None and not about_subject(q, read, found):
            # Read after a dig no source ties to the subject (a wrong hint's
            # name, say): it answers nothing that was asked.
            return None
        return found[0], (*(via or ()), found[1])

    def via(
        self, question: str, read: list[Hit], found: tuple[str, str | None]
    ) -> tuple[str, ...] | None:
        """The source tying the subject to what the answer is about: a name a
        sentence about the subject brings in, named again where the answer is
        said. A chain cites its link; a one-hop answer, or a link in the
        answer's own source, nothing more."""
        said = answer_sentences(read, found)
        for name, source in links(question, read):
            if name != found[0] and any(re.search(rf"\b{re.escape(name)}\b", s) for s in said):
                return () if source == found[1] else (source,)
        return None

    def next_name(self, question: str, read: list[Hit]) -> tuple[str, str] | None:
        """A name the core points at that the question does not hold: one to
        search. A number, or a name already asked about, leads nowhere."""
        found = self.reader.bridge(question, read)  # the Read core's second pointer
        if not found or found[1] is None or found[0] in question:
            return None
        # A capitalised template word ("The", "Records") opens a sentence, it
        # names nothing: searched, it would match half the sources.
        if not found[0][:1].isupper() or set(found[0].split()) & TEMPLATE_WORDS:
            return None
        return found[0], found[1]

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        q, read = framing.question, list(evidence)
        if self.core.prob_enough(q, read) >= self.threshold and self.answer(framing, evidence):
            return Verdict(True)
        if not compared(q) and self.next_name(q, read):
            # Back to Plan: search the name pointed at. After the dig, Plan has
            # nothing more and the round limit ends it as not found.
            return Verdict(False)
        return Verdict(False, Exit.NOT_FOUND)

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        q = framing.question
        if round_ == 0:
            return [q]
        # One dig, never for a comparison: its options are read one by one.
        found = self.next_name(q, list(evidence)) if round_ == 1 and not compared(q) else None
        return [found[0]] if found else []


def answer_sentences(read: list[Hit], found: tuple[str, str | None]) -> list[str]:
    """The sentences of the answer's source that state it."""
    body = next((h.body for h in read if h.source_id == found[1]), "")
    word = rf"\b{re.escape(found[0])}\b"
    return [s for s in re.split(r"(?<=\.)\s+", body) if re.search(word, s)]


def about_subject(question: str, read: list[Hit], found: tuple[str, str | None]) -> bool:
    """The answer stated next to a name the question asks about: a one-hop answer."""
    asked = set(capitalised_words(question))
    return any(asked & set(capitalised_words(s)) for s in answer_sentences(read, found))
