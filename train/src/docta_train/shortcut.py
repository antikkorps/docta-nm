"""Question shape: what the label would look like to a core that never reads.

A single search is enough far more often for some question templates than for
others (a chain question rarely finds both hops at once). A core can then score
well by recognising the template and ignoring the excerpts. The shape is the
question with its names and numbers hidden, so it is the template a core could
learn; balancing labels within each shape removes that shortcut from training,
and `shape_accuracy` measures what it would have been worth.
"""

import random
import re
from collections import Counter, defaultdict
from collections.abc import Sequence

from docta_train.evaluate_data import Example

# Invented names are the only capitalised words besides a question's first
# one, which is hidden too: two templates that differed only by their opening
# word would share a shape (none do today; see worldgen's phrasings.rs).
NAME = re.compile(r"\b[A-Z][a-z]+\b")
NUMBER = re.compile(r"\d+")


def shape(question: str) -> str:
    return NUMBER.sub("#", NAME.sub("_", question))


def balance(examples: Sequence[Example], seed: int) -> list[Example]:
    """Per shape, as many enough as not enough: the minority label is kept
    whole, the majority one sampled down. Shapes with a single label go, since
    their label is the shape. Order is kept so results do not move otherwise."""
    rng = random.Random(seed)
    groups: dict[tuple[str, bool], list[int]] = defaultdict(list)
    for i, e in enumerate(examples):
        groups[(shape(e.question), e.label)].append(i)
    kept: list[int] = []
    for s in sorted({s for s, _ in groups}):
        yes, no = groups[(s, True)], groups[(s, False)]
        n = min(len(yes), len(no))
        kept += rng.sample(yes, n) + rng.sample(no, n)
    return [examples[i] for i in sorted(kept)]


def shape_accuracy(train: Sequence[Example], val: Sequence[Example]) -> float:
    """Accuracy on `val` of the majority label of each shape seen in `train`."""
    if not train or not val:
        raise ValueError("no example to measure the shape shortcut on")
    by_shape: dict[str, Counter[bool]] = defaultdict(Counter)
    for e in train:
        by_shape[shape(e.question)][e.label] += 1
    overall = Counter(e.label for e in train).most_common(1)[0][0]
    guess = {s: c.most_common(1)[0][0] for s, c in by_shape.items()}
    right = sum(guess.get(shape(e.question), overall) == e.label for e in val)
    return right / len(val)
