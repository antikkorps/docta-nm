"""RandomModel: random queries, random verdicts, random answers.

Each decision draws from an RNG seeded by (seed, question, step), so outcomes
do not depend on the order questions are run in.
"""

import random
import re

from docta_orchestrator.states import (
    Conclusion,
    Confrontation,
    Evidence,
    Exit,
    Framing,
    Verdict,
)


class RandomModel:
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def _rng(self, framing: Framing, step: str) -> random.Random:
        return random.Random(f"{self.seed}|{framing.question}|{step}")

    def frame(self, question: str) -> Framing:
        return Framing(question)

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        words = re.findall(r"\w+", framing.question)
        rng = self._rng(framing, f"plan{round_}")
        return rng.sample(words, k=min(2, len(words)))

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        x = self._rng(framing, f"evaluate{len(evidence)}").random()
        return Verdict(False, Exit.NOT_FOUND) if x < 0.3 else Verdict(x < 0.7)

    def confront(self, framing: Framing, evidence: Evidence) -> Confrontation:
        return Confrontation()

    def conclude(self, framing: Framing, evidence: Evidence, confrontation) -> Conclusion:
        rng = self._rng(framing, "conclude")
        text = " ".join(h.body for h in evidence) or framing.question
        ids = [h.source_id for h in evidence]
        cited = rng.sample(ids, k=min(len(ids), rng.randint(1, 2)))
        answer = rng.choice(re.findall(r"\w+", text))
        return Conclusion(answer, tuple(cited), round(rng.random(), 3))
