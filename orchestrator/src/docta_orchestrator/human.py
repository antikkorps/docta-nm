"""The human channel: a person steps into the reflection mid-way.

A hint ("Orlan Teste lives in Varnelle.") arrives as a statement to check, not
a fact: Plan reads it, so it may steer the next search; Evaluate, Confront and
Conclude never do, so the answer must still stand on sources. A wrong hint can
then cost a search, never a wrong answer. The orchestrator enforces this, not
the model.

In the evaluation protocol the simulated human's hint comes from ground truth
(the bridge of a chain, or a wrong one): the one place ground truth reaches a
model, through this channel, at evaluation only, never in training.
"""

from docta_orchestrator.search import Hit

HUMAN = "human"  # the source id of a hint: no source carries it
HINT_ROUND = 1  # Plan's round index the hint reaches: after one search


def human_hint(text: str) -> Hit:
    return Hit(HUMAN, "Human", text, 0.0)
