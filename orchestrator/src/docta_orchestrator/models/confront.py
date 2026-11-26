"""Hand-written Confront: who says what about the asked fact, and who is
independent of whom. Two floors for a trained Confront.

`majority` counts one vote per source: copies of one false origin outvote a
single true source. That is how false consensus fools a vote, and how
repetition fools a model that learned from frequency. `challenge` counts a
source that credits another ("According to Notes on Varnelle, …") together
with that origin, weighs each independent group by its reliability, and
refuses to settle when the two best values are too close.

Both read sentences with the seen templates, like `rule_evaluate`; a copy is
recognised by the title it credits, not by its wording, so unseen credit
phrasings in ood are still caught. Questions they cannot read fall back to
the Evaluate rule and keyword.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from docta_orchestrator.models.rule_evaluate import (
    RuleEvaluateModel,
    needs,
    read_sentence,
    sentences,
)
from docta_orchestrator.states import Confrontation, Evidence, Exit, Framing, Verdict

# Source titles read "Chronicle of X", "Notes on X"…: a credit names one.
TITLE = re.compile(r"\b[A-Z][a-z]+ (?:of|on) [A-Z][a-z]+(?: [A-Z][a-z]+)*")
CREDIT = re.compile(r", | that ")
# Within this reliability gap, two independent sources cannot be told apart.
# Inclusive: the generator keeps undecidable pairs within it and every other
# trap at least 0.10 apart; 1e-9 absorbs float error on an exact 0.05.
TOO_CLOSE = 0.05


@dataclass(frozen=True)
class Claim:
    value: str
    source_id: str
    title: str
    reliability: float
    credits: str | None  # title of the source this claim is repeated from


@dataclass(frozen=True)
class Settled:
    value: str
    cited: tuple[str, ...]


def claims(question: str, evidence: Evidence) -> list[Claim]:
    """Every value the excerpts state for the one fact a retrieve question asks."""
    wanted = needs(question)
    if wanted is None or len(wanted) != 1 or len(wanted[0][1]) != 1:
        return []
    fact = (wanted[0][0], wanted[0][1][0])
    found = []
    for hit in evidence:
        previous = None  # a pronoun means the subject of the sentence before
        for sentence in sentences(hit.body):
            stated, credits = read_claim(sentence, previous)
            previous = stated[0][0] if stated else None
            found += [
                Claim(o, hit.source_id, hit.title, hit.reliability, credits)
                for s, p, o in stated
                if (s, p) == fact
            ]
    return found


def read_claim(sentence: str, previous: str | None = None) -> tuple[list, str | None]:
    """What a sentence states, and the title it credits if it repeats one.
    The credit is looked for first: read whole, "As T reports, Brask is home to
    X." fits "{o} is home to {s}." with the credit swallowed into the object."""
    for split in CREDIT.finditer(sentence):
        rest = sentence[split.end() :]
        title = TITLE.search(sentence[: split.start()])
        stated = read_sentence(rest[:1].upper() + rest[1:])
        if stated and title:
            return stated, title.group()
    return read_sentence(sentence, previous), None


def vote(found: list[Claim]) -> Settled:
    """One voice per source, however often it repeats itself; reliability
    only breaks ties."""
    one_each = list({c.source_id: c for c in reversed(found)}.values())
    tally: dict[str, tuple[int, float]] = {}
    for c in one_each:
        count, weight = tally.get(c.value, (0, 0.0))
        tally[c.value] = (count + 1, weight + c.reliability)
    value = max(sorted(tally), key=lambda v: tally[v])
    return Settled(value, tuple(c.source_id for c in found if c.value == value))


def weigh(found: list[Claim]) -> Settled | None:
    """Copies join their origin; each independent group weighs as its origin
    (or its best copy, if the origin was not read). None: too close to call."""
    groups: dict[tuple[str, str], list[Claim]] = {}
    for c in found:
        groups.setdefault((origin_of(c, found), c.value), []).append(c)
    weight: dict[str, float] = {}
    cited: dict[str, list[str]] = {}
    for (_, value), group in groups.items():
        origins = [c for c in group if c.credits is None] or group
        best = max(origins, key=lambda c: c.reliability)
        weight[value] = weight.get(value, 0.0) + best.reliability
        cited.setdefault(value, []).append(best.source_id)
    ranked = sorted(weight, key=lambda v: (-weight[v], v))
    if len(ranked) > 1 and weight[ranked[0]] - weight[ranked[1]] <= TOO_CLOSE + 1e-9:
        return None
    return Settled(ranked[0], tuple(cited[ranked[0]]))


def origin_of(c: Claim, found: list[Claim]) -> str:
    """The independent source behind a claim: itself (by id, as true sources
    may share a title), or the origin a copy credits, by id if it was read.
    Among read sources sharing the credited title, the origin is the one
    saying what the copy repeats: a copy never adds a voice of its own."""
    if c.credits is None:
        return c.source_id
    named = [o for o in found if o.credits is None and o.title == c.credits]
    same = [o for o in named if o.value == c.value]
    return (same or named)[0].source_id if named else c.credits


class ConfrontRule(RuleEvaluateModel, ABC):
    """The Evaluate rule, plus a Confront that settles disagreements."""

    @abstractmethod
    def decide(self, found: list[Claim]) -> Settled | None: ...

    def trusts(self, found: list[Claim]) -> bool:
        """Whether what was read is worth weighing at all."""
        return True

    def evaluate(self, framing: Framing, evidence: Evidence) -> Verdict:
        found = claims(framing.question, evidence)
        if not found:
            return super().evaluate(framing, evidence)
        # The fact is stated: Confront weighs who states what.
        return Verdict(True) if self.trusts(found) else Verdict(False, Exit.NOT_FOUND)

    def confront(self, framing: Framing, evidence: Evidence) -> Confrontation:
        found = claims(framing.question, evidence)
        if found and self.decide(found) is None:
            return Confrontation(exit=Exit.CONTRADICTION)
        return Confrontation()

    def answer(self, framing: Framing, evidence: Evidence):
        found = claims(framing.question, evidence)
        if not found:
            return super().answer(framing, evidence)
        settled = self.decide(found)
        return (settled.value, settled.cited) if settled else None


class MajorityModel(ConfrontRule):
    name = "majority"

    def decide(self, found: list[Claim]) -> Settled | None:
        return vote(found)


class ChallengeModel(ConfrontRule):
    name = "challenge"

    def decide(self, found: list[Claim]) -> Settled | None:
        return weigh(found)

    def trusts(self, found: list[Claim]) -> bool:
        # Copies alone are hearsay: without one independent source, doubt.
        return any(c.credits is None for c in found)
