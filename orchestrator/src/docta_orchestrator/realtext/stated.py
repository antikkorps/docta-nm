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

# A capital after the stop keeps "c. 1750" whole.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# A first name then a lone capital is an initial ("John G. Kemeny"): the hand
# check found links lost to that cut. After a lowercase word it is not ("in C.").
_INITIAL = re.compile(r"\b[A-Z][a-z]+ [A-Z]\.$")
_SUFFIXES = {"Jr.", "Sr.", "Jr", "Sr", "II", "III", "IV"}
_NUMBER = re.compile(r"\b\d+\b")
MIN_NAME = 3
# Words a sentence uses to state a chain's first hop.
LINK_WORDS = {"born_in": ("born",), "designed_by": ("design", "creat", "develop", "invent")}
# Which of a link's two entities is the person: an article may name them by
# surname alone ("Kemeny and Kurtz").
PERSON = {"born_in": 0, "designed_by": 1}


def paragraphs(article: dict) -> Iterable[str]:
    return (p for section in article["sections"] for p in section["paragraphs"])


def sentences(text: str) -> list[str]:
    out: list[str] = []
    for part in _SENTENCE_END.split(text):
        if out and _INITIAL.search(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        elif part:
            out.append(part)
    return out


def mentions(text: str, word: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(word)}(?!\w)", text) is not None


def variants(person: str) -> list[str]:
    """A person's label, and their surname when it is long enough to be a
    name rather than an initial."""
    words = [w for w in person.replace(",", " ").split() if w not in _SUFFIXES]
    surname = words[-1] if len(words) > 1 else ""
    return [person, surname] if len(surname) >= MIN_NAME else [person]


class Index:
    """Each article's sentences by subject, and every sentence by the
    numbers it holds, so a check reads a handful of sentences, not the
    corpus. Sentences are split per paragraph, so a redaction can find each
    one again where it stands."""

    def __init__(self, articles: Iterable[dict]):
        self.own: dict[str, list[str]] = {}
        self.by_number: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for a in articles:
            self.own[a["qid"]] = [s for p in paragraphs(a) for s in sentences(p)]
            for sentence in self.own[a["qid"]]:
                for number in set(_NUMBER.findall(sentence)):
                    self.by_number[number].append((a["qid"], sentence))

    def stating(self, qid: str, label: str, values: Iterable) -> list[tuple[str, str]]:
        """Every (article, sentence) that states one of the values for the
        subject: its own article's, or one elsewhere naming it. A label of one
        or two letters ("J", "C") matches initials and "C++", so only the
        subject's own article can speak for it."""
        out = []
        for value in map(str, values):
            out += [(qid, s) for s in self.own.get(qid, []) if mentions(s, value)]
            if len(label) >= MIN_NAME:
                out += [(q, s) for q, s in self.by_number.get(value, ())
                        if q != qid and mentions(s, label)]  # fmt: skip
        return out

    def stated(self, qid: str, label: str, values: Iterable) -> str | None:
        """The first value the texts state for the subject, if any."""
        return next((str(v) for v in values if self.stating(qid, label, [v])), None)

    def states(self, qid: str, label: str, values: Iterable) -> bool:
        return self.stated(qid, label, values) is not None

    def linking(self, a: tuple[str, str], b: tuple[str, str], link: str) -> list[tuple[str, str]]:
        """Every (article, sentence) in either article of two entities (qid,
        label) that names the other and says the link: "born in Lincoln" ties
        a person to a city, "University of Nebraska–Lincoln" does not."""
        names = [[a[1]], [b[1]]]
        names[PERSON[link]] = variants(names[PERSON[link]][0])
        out = []
        for (qid, _), others in ((a, names[1]), (b, names[0])):
            for sentence in self.own.get(qid, []):
                low = sentence.lower()
                named = any(mentions(sentence, n) for n in others)
                if named and any(w in low for w in LINK_WORDS[link]):
                    out.append((qid, sentence))
        return out

    def links(self, a: tuple[str, str], b: tuple[str, str], link: str) -> bool:
        return bool(self.linking(a, b, link))
