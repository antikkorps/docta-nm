"""AlwaysAbstainModel: says "not found" to everything, without searching."""

from docta_orchestrator.states import (
    Confrontation,
    Evidence,
    Exit,
    Framing,
    Verdict,
)


class AlwaysAbstainModel:
    name = "always_abstain"

    def frame(self, question: str) -> Framing:
        return Framing(question)

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        return []

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        return Verdict(False, Exit.NOT_FOUND)

    def confront(self, framing: Framing, evidence: Evidence) -> Confrontation:
        raise AssertionError("unreachable: evaluate always exits")

    def conclude(self, framing, evidence, confrontation):
        raise AssertionError("unreachable: evaluate always exits")
