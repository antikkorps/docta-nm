"""Name marks: which bytes of what the core reads repeat a name of the question.

Matching an invented name across the question and the excerpts is computable,
so deterministic code does it, as the project asks of anything computable. A
byte-level core otherwise spends its capacity learning to compare spellings,
and the first one never got past guessing from the question's shape. The core
still judges whether what it read is enough; it no longer has to find the names.
"""

import re

from docta_train.shortcut import NAME

MAX_NAMES = 4  # a comparison of two people names four words


def names(question: str) -> list[str]:
    """Capitalised words of the question except its opening one ("Who", "In").

    Relies on worldgen: every question template opens with a question word, and
    names are ASCII syllables, which is all NAME matches."""
    question = question.strip()
    return [m.group() for m in NAME.finditer(question) if m.start() > 0]


def name_marks(question: str, text: str) -> list[int]:
    """One mark per UTF-8 byte of `text`: k inside a whole-word occurrence of
    the k-th distinct name word of `question`, 0 elsewhere."""
    ids: dict[str, int] = {}
    for word in names(question):
        ids.setdefault(word, min(len(ids) + 1, MAX_NAMES))
    chars = [0] * len(text)
    for word, k in ids.items():
        for m in re.finditer(rf"\b{re.escape(word)}\b", text):
            chars[m.start() : m.end()] = [k] * (m.end() - m.start())
    return [k for ch, k in zip(text, chars, strict=True) for _ in ch.encode()]
