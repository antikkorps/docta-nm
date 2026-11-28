"""Which words are names, and what replaces them.

No trained model may find the names, so the corpus decides by counting: a
word is a name when the domain writes it capitalised in mid-sentence more
often than in lowercase ("China" a hundred times, "china" once; "bridge"
far more often than "Bridge"). A word of the answer key or of a link that
is never written in lowercase counts too, even at sentence starts only.
Sentence starts alone prove nothing: every word is capitalised there."""

import re
from collections import Counter
from collections.abc import Iterable

# A run of letters, any script ("Köprüsü"), with the combining marks that
# Devanagari, Arabic, Hebrew or Thai write inside a word, so "हिन्दी" stays
# one word instead of three letters between orphan vowel signs.
MARKS = (
    "\u0300-\u036f\u0483-\u0489\u0591-\u05c7\u0610-\u061a\u064b-\u065f\u0670"
    "\u06d6-\u06ed\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962\u0963"
    "\u0981-\u0983\u09bc-\u09d7\u0e31\u0e34-\u0e3a\u0e47-\u0e4e"
)
WORD = re.compile(rf"[^\W\d_](?:[^\W\d_]|[{MARKS}])*")


def capitalised(text: str) -> set[str]:
    return {w for w in WORD.findall(text) if len(w) > 1 and w[0].isupper()}


def texts(article: dict) -> Iterable[str]:
    for section in article["sections"]:
        yield from section["path"]
        yield from section["paragraphs"]


def named(entity: dict) -> Iterable[str]:
    """Every string the answer key names: label, title, aliases, related."""
    yield entity["label"]
    yield entity["title"]
    yield from entity.get("aliases", [])
    for values in entity["facts"].values():
        yield from (v["label"] for v in values if isinstance(v, dict) and "label" in v)


# Capitalised by the language, not because they name something; a short
# fixed list, like the templates' words.
CALENDAR = {
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}  # fmt: skip
# Small Roman numerals only (kings, wars, volumes): "DC" or "MD" are names.
# Capitals the language writes: eras, times of day, units.
CAPITALS = {"BC", "AD", "BCE", "CE", "AM", "PM", "MW", "GW", "KW", "GB", "MB", "TB", "OK", "TV"}
ROMAN = re.compile(r"X{0,3}(IX|IV|V?I{0,3})")
# Abbreviations whose full stop ends no sentence: "Mt. Hood", "U.S. Navy".
ABBREVIATIONS = {
    "St", "Mt", "Mr", "Mrs", "Ms", "Dr", "Ft", "Jr", "Sr", "No", "vs", "Prof",
    "Gen", "Col", "Lt", "Capt", "Rev", "Gov", "Sen", "Rep", "Co", "Corp", "Inc",
}  # fmt: skip
OPENERS = "\"'“‘«([ \t\n"


def starts_sentence(before: str) -> bool:
    """True when a word preceded by `before` opens a sentence: at the start
    of a text, after an opening quote, or after . ! ? that is no
    abbreviation's or initial's full stop."""
    rest = before.rstrip(OPENERS)
    if not rest:
        return True
    if rest[-1] in "!?":
        return True
    if rest[-1] != ".":
        return False
    last = re.split(r"\s", rest[:-1])[-1]
    abbreviated = last in ABBREVIATIONS or re.fullmatch(r"(?:[A-Z]\.)*[A-Z]", last)
    return not abbreviated


def is_excepted(word: str) -> bool:
    """Capitalised by the language or an abbreviation, never a name."""
    return (word in CALENDAR or word in ABBREVIATIONS or word in CAPITALS
            or bool(ROMAN.fullmatch(word)))  # fmt: skip


def lowercase_words(articles: list[dict]) -> set[str]:
    """Every word written in lowercase somewhere: common words of the corpus."""
    return {w for a in articles for t in texts(a) for w in WORD.findall(t) if w.islower()}


def counts(articles: list[dict]) -> tuple[Counter, Counter]:
    """How often each word is written in lowercase, and capitalised in
    mid-sentence (not at the start of a text or after . ! ?)."""
    low, mid = Counter(), Counter()
    for t in (t for a in articles for t in texts(a)):
        for m in WORD.finditer(t):
            w = m.group(0)
            if w.islower():
                low[w] += 1
            elif w[0].isupper() and not starts_sentence(t[: m.start()]):
                mid[w] += 1
    return low, mid


def always_names(articles: list[dict]) -> set[str]:
    """Words whose form alone makes them names: scripts without letter case
    (이순신대교, 積石山), where capitals prove nothing, and acronyms in
    capitals ("SKY"), judged as written, not by their lowercase twin."""
    found = set()
    for w in (w for a in articles for t in texts(a) for w in WORD.findall(t)):
        caseless = w.lower() == w.upper() and w[0].isalpha()  # not "½", which \\w lets in
        if caseless or (len(w) > 1 and w.isupper()):
            found.add(w)
    return found


def entity_words(articles: list[dict], entities: list[dict]) -> set[str]:
    """Words the corpus treats as names (see the module's docstring)."""
    sources = [t for e in entities for t in named(e)]
    sources += [t for a in articles for t in a.get("links", [])]
    keyed = set().union(*(capitalised(t) for t in sources))
    low, mid = counts(articles)
    found = {w for w in keyed if low[w.lower()] == 0}
    found |= {w for w, n in mid.items() if len(w) > 1 and low[w.lower()] < n}
    found |= always_names(articles)
    return {w for w in found if not is_excepted(w)}


class Namer:
    """Hands out invented names from a pool, skipping any the corpus already
    writes as a word, so a replacement never reads as an existing word."""

    def __init__(self, pool: list[str], forbidden: set[str]):
        self.pool, self.forbidden = pool, forbidden

    def assign(self, originals: set[str]) -> dict[str, str]:
        """Sorted originals take pool names in order: the same corpus maps the
        same way whatever order its articles came in."""
        free = (n for n in self.pool if n.lower() not in self.forbidden)
        out = {}
        for original in sorted(originals):
            name = next(free, None)
            if name is None:
                raise ValueError(f"name pool too small for {len(originals)} names")
            out[original] = name
        return out
