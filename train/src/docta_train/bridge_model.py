"""CoreBridgeModel: core_read with no rule on a chain's wording.

"In what year was the city Orlan Teste resides in founded?" is read whole.
The Read core points at what the question needs next: the answer once its
source was read, otherwise the name to search (Varnelle, from "Orlan Teste
resides in Varnelle."). Code searches that name, as dig does, and the core
points again. The Evaluate core judges the whole question on all that was
read. No regex recognises a chain, so a held-out link wording is no wall:
whether the core finds the bridge is what ood measures.

A human hint reaches Plan only: the core may point at a name the hint says,
which steers the search, but the answer and its citations stand on sources.
"""

import re

from docta_orchestrator.models.keyword import Found, compared
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
        return found[0], (*self.via(q, read, found), found[1])

    def via(self, question: str, read: list[Hit], found: tuple[str, str | None]) -> tuple[str, ...]:
        """The source that led to the answer's: what the core points at with
        the answer's source set aside, if that name is what the answer's
        source is about. A chain cites its link; a one-hop answer, nothing more."""
        rest = [h for h in read if h.source_id != found[1]]
        bridge = self.next_name(question, rest)
        if bridge is None or bridge[0] == found[0]:
            return ()  # the answer met again elsewhere is no bridge to it
        about = next((h.body for h in read if h.source_id == found[1]), "")
        return (bridge[1],) if re.search(rf"\b{re.escape(bridge[0])}\b", about) else ()

    def next_name(self, question: str, read: list[Hit]) -> tuple[str, str] | None:
        """A name the core points at that the question does not hold: one to
        search. A number, or a name already asked about, leads nowhere."""
        found = self.reader.read(question, read)
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
