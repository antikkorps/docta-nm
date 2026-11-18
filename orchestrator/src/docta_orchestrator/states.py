"""The method loop's vocabulary: states, emergency exits, and what a model fills in.

The orchestrator owns the transitions; a model only fills each state. That
keeps the method traceable and testable before any of it moves into weights.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from docta_orchestrator.search import Hit


class State(StrEnum):
    FRAME = "frame"
    PLAN = "plan"
    CONSULT = "consult"
    EVALUATE = "evaluate"
    CONFRONT = "confront"
    CONCLUDE = "conclude"
    DONE = "done"


class Exit(StrEnum):
    """Emergency exits: part of the method, not failures."""

    ILL_POSED = "ill_posed_question"
    NOT_FOUND = "not_found"
    CONTRADICTION = "unresolved_contradiction"


@dataclass(frozen=True)
class Framing:
    """Frame: the question restated, and what is unknown about it."""

    question: str
    unknowns: tuple[str, ...] = ()
    exit: Exit | None = None


@dataclass(frozen=True)
class Verdict:
    """Evaluate: is the evidence enough to answer?"""

    enough: bool
    exit: Exit | None = None


@dataclass(frozen=True)
class Confrontation:
    """Confront: what the excerpts agree on, or why they cannot be reconciled."""

    agreements: tuple[str, ...] = ()
    exit: Exit | None = None


@dataclass(frozen=True)
class Conclusion:
    answer: str
    cited: tuple[str, ...]
    confidence: float


Evidence = tuple[Hit, ...]


class Model(Protocol):
    """What the future trained core implements; phase 0 has fake models."""

    name: str

    def frame(self, question: str) -> Framing: ...

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        """Queries to run next; an empty list means nothing more to search."""
        ...

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict: ...

    def confront(self, framing: Framing, evidence: Evidence) -> Confrontation: ...

    def conclude(
        self, framing: Framing, evidence: Evidence, confrontation: Confrontation
    ) -> Conclusion: ...
