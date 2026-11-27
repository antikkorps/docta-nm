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
MIN_NAME = 3
# Words a sentence uses to state a chain's first hop.
LINK_WORDS = {"born_in": ("born",), "designed_by": ("design", "creat", "develop", "invent")}


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

    def stated(self, qid: str, label: str, values: Iterable) -> str | None:
        """The first value the texts state for the subject, if any."""
        own = self.own.get(qid, "")
        for value in map(str, values):
            if mentions(own, value):
                return value
            # A label of one or two letters ("J", "C") matches initials and
            # "C++": only the subject's own article can speak for it.
            if len(label) >= MIN_NAME and any(
                mentions(s, label) for s in self.by_number.get(value, ())
            ):
                return value
        return None

    def states(self, qid: str, label: str, values: Iterable) -> bool:
        return self.stated(qid, label, values) is not None

    def links(self, a: tuple[str, str], b: tuple[str, str], link: str) -> bool:
        """Whether a sentence in either article of two entities (qid, label)
        names the other and says the link: "born in Lincoln" ties a person to
        a city, "University of Nebraska–Lincoln" does not."""
        for (qid, _), (_, other) in ((a, b), (b, a)):
            for sentence in sentences(self.own.get(qid, "")):
                low = sentence.lower()
                if mentions(sentence, other) and any(w in low for w in LINK_WORDS[link]):
                    return True
        return False
