"""Whether the texts state a value for a subject: the line between an
answerable question and one the core must abstain on.

A year is almost always somewhere in two million words, so absence from the
whole corpus means nothing: a value counts as stated when the subject's own
article holds it, or when a sentence elsewhere holds it next to the
subject's name.
"""

import re
from collections import defaultdict
from collections.abc import Iterable

# A capital after the stop keeps "c. 1750" and "St. Louis" whole.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_NUMBER = re.compile(r"\b\d+\b")


def paragraphs(article: dict) -> Iterable[str]:
    return (p for section in article["sections"] for p in section["paragraphs"])


def sentences(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text) if s]


def mentions(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


class Index:
    """Each article's text by subject, and every sentence by the numbers it
    holds, so a check reads a handful of sentences, not the corpus."""

    def __init__(self, articles: Iterable[dict]):
        self.own: dict[str, str] = {}
        self.by_number: dict[str, list[str]] = defaultdict(list)
        for a in articles:
            text = " ".join(paragraphs(a))
            self.own[a["qid"]] = text
            for sentence in sentences(text):
                for number in set(_NUMBER.findall(sentence)):
                    self.by_number[number].append(sentence)

    def states(self, qid: str, label: str, values: Iterable) -> bool:
        own = self.own.get(qid, "")
        for value in map(str, values):
            if mentions(own, value):
                return True
            if any(mentions(s, label) for s in self.by_number.get(value, ())):
                return True
        return False

    def links(self, a: tuple[str, str], b: tuple[str, str]) -> bool:
        """Whether either article of two entities (qid, label) names the
        other: the first hop of a chain (a birthplace, a designer)."""
        return mentions(self.own.get(a[0], ""), b[1]) or mentions(self.own.get(b[0], ""), a[1])
