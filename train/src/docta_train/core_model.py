"""CoreEvaluateModel: keyword's plan and conclusion, the trained core's judgement.

Only Evaluate is learned, so any change against `keyword` comes from the core
deciding whether what was read is enough.
"""

from typing import Protocol

from docta_orchestrator.models import KeywordModel
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
