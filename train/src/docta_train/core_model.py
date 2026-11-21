"""Models whose Evaluate state is the trained core.

CoreEvaluateModel: keyword's plan and conclusion, the core's judgement. Only
Evaluate is learned, so any change against `keyword` comes from the core
deciding whether what was read is enough.

CoreChainModel: dig's plan and conclusion; the orchestrator splits a chain
into one-hop questions and the core judges each, since read whole a chain
binds two facts in one reading, which the core fails at.
"""

from typing import Protocol

from docta_orchestrator.models import KeywordModel
from docta_orchestrator.models.decompose import Chain, bridges, split_chain
from docta_orchestrator.models.dig import DigModel, can_dig, dug_answer
from docta_orchestrator.models.keyword import Found
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Evidence, Exit, Framing, Verdict


class Core(Protocol):
    def prob_enough(self, question: str, excerpts: list[Hit]) -> float: ...


class CoreEvaluateModel(KeywordModel):
    name = "core_evaluate"

    def __init__(self, core: Core, threshold: float = 0.5):
        self.core = core
        self.threshold = threshold

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        judged_enough = self.core.prob_enough(framing.question, list(evidence)) >= self.threshold
        # Enough by the core, but no answer to read: nothing may be stated unsupported.
        if judged_enough and self.answer(framing, evidence) is not None:
            return Verdict(True)
        return Verdict(False, Exit.NOT_FOUND)


def passing_bridges(chain: Chain, evidence: Evidence, core: Core, threshold: float) -> set[str]:
    """Bridges that pass both hops. The first hop reads only the sentences
    revealing the bridge: on every excerpt, a guild would pass for the city
    the question asks about."""
    passing = set()
    for bridge, revealing in bridges(chain, evidence).items():
        first = core.prob_enough(chain.first_hop, list(revealing)) >= threshold
        if first and core.prob_enough(chain.second_hop(bridge), list(evidence)) >= threshold:
            passing.add(bridge)
    return passing


class CoreChainModel(DigModel):
    name = "core_chain"

    def __init__(self, core: Core, threshold: float = 0.5):
        self.core = core
        self.threshold = threshold

    def answer(self, framing: Framing, evidence: Evidence) -> Found | None:
        """A chain's answer is read about a bridge the core accepted, never
        about another name the subject's sources bring in."""
        chain = split_chain(framing.question)
        if chain is None:
            return super().answer(framing, evidence)
        accepted = passing_bridges(chain, evidence, self.core, self.threshold)
        return dug_answer(framing.question, evidence, accepted) if accepted else None

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        q = framing.question
        chain = split_chain(q)
        # For a chain, answer() already holds the core's verdict on each hop.
        judged = chain is not None or self.core.prob_enough(q, list(evidence)) >= self.threshold
        # Enough by the core, but no answer to read: nothing may be stated unsupported.
        if judged and self.answer(framing, evidence) is not None:
            return Verdict(True)
        if chain is not None and can_dig(q, evidence):
            return Verdict(False)  # back to Plan: search the revealed bridge
        return Verdict(False, Exit.NOT_FOUND)
