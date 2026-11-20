"""Word tokens for the Evaluate core: the templates' words, and names by rank.

Read byte by byte, the core never learned to tie "population" in a question to
"inhabitants" in an excerpt: even single-hop questions stayed near chance.
Words make that tie a matter of a few embeddings. The vocabulary comes from the
training split only, never from an outside file: a word is kept when it shows
up in enough worlds, which the templates' words do and invented names never
do (on 20,000 worlds, 83 words and 3 signs reach 20 worlds, no name does: 97
tokens with the special ones). Worldgen's name exclusion list must keep names
apart from those words, or a name such as "Was" would read as `was`. A name of the
question becomes its rank there, so the core sees where each one recurs without
ever reading its spelling. Any other name gets a rank by order of appearance
(<other_k>): a chain holds only if the city one excerpt says someone lives in
is the city another excerpt dates, and a single <name> token hid that.
"""

import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

from docta_orchestrator.search import Hit

from docta_train.evaluate_data import Example
from docta_train.shortcut import NAME

SPECIAL = ("<pad>", "<cls>", "<unk>", "<num>", "<name>", "<src>", "<body>")
MAX_NAMES = 4  # a comparison of two people names four words
RANKS = tuple(f"<name_{k}>" for k in range(1, MAX_NAMES + 1))
MAX_OTHERS = 24  # an example names at most 22 words outside its question
OTHERS = tuple(f"<other_{k}>" for k in range(1, MAX_OTHERS + 1))
FIXED = SPECIAL + RANKS + OTHERS
PAD = 0
WORD = re.compile(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]")


def names(question: str) -> list[str]:
    """Capitalised words of the question except its opening one ("Who", "In").

    Relies on worldgen: every question template opens with a question word, and
    names are ASCII syllables with one capital, which both NAME and WORD match."""
    question = question.strip()
    return [m.group() for m in NAME.finditer(question) if m.start() > 0]


class Vocab:
    def __init__(self, words: Sequence[str]):
        if tuple(words[: len(FIXED)]) != FIXED:
            raise ValueError("vocabulary without the current special tokens")
        self.words = tuple(words)
        self.index = {w: i for i, w in enumerate(self.words)}

    @classmethod
    def build(cls, examples: Iterable[Example], min_worlds: int = 20) -> "Vocab":
        """Lowercased words and signs seen in at least `min_worlds` worlds."""
        worlds: dict[str, set[str]] = defaultdict(set)
        for e in examples:
            text = " ".join([e.question, *(f"{h.title} {h.body}" for h in e.excerpts)])
            for w in WORD.findall(text):
                worlds[w.lower()].add(e.world_id)
        kept = sorted(w for w, seen in worlds.items() if len(seen) >= min_worlds)
        return cls(FIXED + tuple(w for w in kept if not w.isdigit()))

    def tokens(self, question: str, excerpts: Iterable[Hit]) -> list[str]:
        """`ranks` grows as other names are met, so each keeps its token."""
        ranks: dict[str, str] = {}
        for name in names(question):
            ranks.setdefault(name, RANKS[min(len(ranks), MAX_NAMES - 1)])
        out = ["<cls>", *self.words_of(question, ranks)]
        for hit in excerpts:
            out += ["<src>", *self.words_of(hit.title, ranks)]
            out += ["<body>", *self.words_of(hit.body, ranks)]
        return out

    def words_of(self, text: str, ranks: dict[str, str]) -> list[str]:
        return [self.token(w, ranks) for w in WORD.findall(text)]

    def token(self, word: str, ranks: dict[str, str]) -> str:
        if word.isdigit():
            return "<num>"
        if word in ranks:
            return ranks[word]
        if word.lower() in self.index:
            return word.lower()
        if not word[0].isupper():
            return "<unk>"
        met = len(ranks) - sum(r in RANKS for r in ranks.values())
        ranks[word] = OTHERS[met] if met < MAX_OTHERS else "<name>"
        return ranks[word]

    def encode(self, question: str, excerpts: Iterable[Hit]) -> list[int]:
        return [self.index[t] for t in self.tokens(question, excerpts)]
