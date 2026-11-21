"""Splits a two-hop question into one-hop questions.

The Evaluate core judges one-hop questions well and chains near chance:
binding a name to the city it lives in, then that city to its founding year,
in one reading is what it fails at. So the orchestrator splits the chain and
the core judges each hop.

"In what year was the city where Orlan Teste lives founded?" becomes:
- first hop, the link question train shows: "In which city does Orlan Teste live?"
- second hop, the question with the bridge in place of the link: "In what
  year was Varnelle founded?". The outer wording is kept on purpose: a held-out
  phrasing stays held out, so any ood gain comes from the split, not from a
  rewrite into seen words.

The cue is the chain wording (CHAIN_LINKS), which every split shares, ood
included, as DigModel already relies on.
"""

import re
from dataclasses import dataclass

from docta_orchestrator.models.dig import proper_names
from docta_orchestrator.models.keyword import capitalised_words
from docta_orchestrator.models.templates import CHAIN_LINKS, QUESTIONS
from docta_orchestrator.search import Hit
from docta_orchestrator.states import Evidence

NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+)*"
BRIDGE = "\0"  # stands for the bridge in the outer question; no template holds it


@dataclass(frozen=True)
class Chain:
    link: str  # the first hop's predicate: lives_in, based_in, member_of
    subject: str
    outer: str  # the question with BRIDGE in place of the link description

    @property
    def first_hop(self) -> str:
        return QUESTIONS[self.link][0].format(s=self.subject)

    def second_hop(self, bridge: str) -> str:
        return self.outer.replace(BRIDGE, bridge)


def split_chain(question: str) -> Chain | None:
    for link, description in CHAIN_LINKS.items():
        pattern = re.escape(description).replace(re.escape("{s}"), f"(?P<s>{NAME})")
        m = re.search(pattern, question)
        if m:
            outer = question[: m.start()] + BRIDGE + question[m.end() :]
            return Chain(link, m.group("s"), outer)
    return None


def bridges(chain: Chain, evidence: Evidence) -> dict[str, tuple[Hit, ...]]:
    """Each name a sentence about the subject brings in, with those sentences
    as excerpts. The first hop is judged on them alone: on whole sources, a
    guild named next to the city would pass for the city asked about."""
    subject = set(chain.subject.split())
    found: dict[str, tuple[Hit, ...]] = {}
    for hit in evidence:
        for sentence in re.split(r"(?<=\.)\s+", hit.body):
            if not subject & set(capitalised_words(sentence)):
                continue
            for name in proper_names(sentence):
                if not subject & set(name.split()):
                    said = Hit(hit.source_id, hit.title, sentence, hit.reliability)
                    found[name] = (*found.get(name, ()), said)
    return found
