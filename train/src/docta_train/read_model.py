"""CoreReadModel: core_chain's judgement, with the Read core's answers.

The Evaluate core judges whether what was read is enough; the Read core
points at the answer and code copies it. Comparisons: the Read core reads
each option's value, code compares. Chains: it reads the bridge on the link
question, which leads the second search, then the value about the bridge.
Only Plan's first query and Confront stay rules.
"""

from typing import Protocol

from docta_orchestrator.models.decompose import Chain, split_chain
from docta_orchestrator.models.dig import dig_query
from docta_orchestrator.models.keyword import Found, compared, winner
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Evidence, Framing

from docta_train.core_model import Core, CoreChainModel

Reading = tuple[str, str | None]  # (answer, source id)


class Reader(Protocol):
    def read(self, question: str, excerpts: list[Hit]) -> Reading | None: ...


class CoreReadModel(CoreChainModel):
    name = "core_read"

    def __init__(self, core: Core, reader: Reader, threshold: float = 0.5):
        super().__init__(core, threshold)
        self.reader = reader

    def answer(self, framing: Framing, evidence: Evidence) -> Found | None:
        q, read = framing.question, list(evidence)
        options = compared(q)
        if options:
            return self.compare(q, options, read)
        chain = split_chain(q)
        if chain is not None:
            return self.through(chain, read)
        found = self.reader.read(q, read)
        return (found[0], (found[1],)) if found else None

    def compare(self, question: str, options: tuple[str, str], read: list[Hit]) -> Found | None:
        # The option is named again after the question, as the core was trained.
        values = [self.reader.read(f"{question} {o}", read) for o in options]
        if any(v is None or not v[0].isdigit() for v in values):
            return None
        (va, sa), (vb, sb) = values
        best = winner(question, (options[0], int(va)), (options[1], int(vb)))
        return best, tuple(dict.fromkeys((sa, sb)))

    def through(self, chain: Chain, read: list[Hit]) -> Found | None:
        """The value read about the bridge, once the Evaluate core judged both
        hops enough: the link on the bridge's source, the value on everything."""
        bridge = self.bridge(chain, read)
        if bridge is None:
            return None
        name, via = bridge
        second = chain.second_hop(name)
        revealing = [h for h in read if h.source_id == via]
        if min(self.core.prob_enough(chain.first_hop, revealing),
               self.core.prob_enough(second, read)) < self.threshold:  # fmt: skip
            return None
        found = self.reader.read(second, read)
        return (found[0], tuple(dict.fromkeys((via, found[1])))) if found else None

    def bridge(self, chain: Chain, read: list[Hit]) -> Reading | None:
        found = self.reader.read(chain.first_hop, read)
        return found if found and not found[0].isdigit() else None  # a bridge is a name

    def digs(self, question: str, evidence: Evidence) -> bool:
        chain = split_chain(question)
        return chain is not None and self.bridge(chain, list(evidence)) is not None

    def plan(self, framing: Framing, evidence: Evidence, round_: int) -> list[str]:
        if round_ != 1:
            return super().plan(framing, evidence, round_)  # the question, then nothing
        chain = split_chain(framing.question)
        bridge = self.bridge(chain, list(evidence)) if chain else None
        return [dig_query(framing.question, bridge[0])] if bridge else []
