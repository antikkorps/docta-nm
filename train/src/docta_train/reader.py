"""The Read core: points at the answer in what was read; code copies it.

The core never writes an answer, so it never needs a fact or even a digit:
it scores each word of the excerpts' bodies, and the orchestrator copies the
number, or the whole name, found at the best one, citing its source. It is
the Evaluate core's encoder with one score per token instead of one per
example: same word tokens, same sentence-local heads.
"""

import torch

from docta_train.model import CoreEvaluate
from docta_train.words import BODY, PAD, SRC, STOP, WORD

Aligned = list[tuple[str, str | None]]  # (word, source id when in a body)


def aligned(question: str, excerpts) -> Aligned:
    """Words in the order Vocab.tokens encodes them, each with the source
    whose body holds it: titles and the question are never an answer."""
    out: Aligned = [("<cls>", None), *((w, None) for w in WORD.findall(question))]
    for hit in excerpts:
        out += [("<src>", None), *((w, None) for w in WORD.findall(hit.title))]
        out += [("<body>", None), *((w, hit.source_id) for w in WORD.findall(hit.body))]
    return out


def answer_positions(words: Aligned, answer: str, sources, max_len: int) -> list[int]:
    """Where the answer's words start in one of the sources stating it."""
    wanted = WORD.findall(answer)
    found = []
    for i, (_, source) in enumerate(words[:max_len]):
        span = words[i : i + len(wanted)]
        if source in sources and [w for w, _ in span] == wanted:
            found.append(i)
    return found


def extract(words: Aligned, at: int) -> tuple[str, str | None]:
    """The number at `at`, or the name starting there with the capitalised
    words that follow it in the same source ("Kel Dorn")."""
    word, source = words[at]
    if word.isdigit():
        return word, source
    name = [word]
    for nxt, where in words[at + 1 :]:
        if where != source or not nxt[:1].isupper() or not nxt.isalpha():
            break
        name.append(nxt)
    return " ".join(name), source


def candidates(ids: torch.Tensor) -> torch.Tensor:
    """Body words: after a <body> and before the next <src>, signs aside."""
    marks = (ids == SRC) | (ids == BODY)
    index = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    last = torch.where(marks, index, torch.zeros_like(index)).cummax(1).values
    in_body = ids.gather(1, last) == BODY
    return in_body & ~marks & (ids != STOP) & (ids != PAD)


def pointer_loss(scores: torch.Tensor, target: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    """-log of the probability, among allowed tokens, of pointing at any
    position of the answer: every occurrence counts as right."""
    low = torch.finfo(scores.dtype).min
    every = scores.masked_fill(~allowed, low).logsumexp(1)
    right = scores.masked_fill(~(target & allowed), low).logsumexp(1)
    return (every - right).mean()


class CoreRead(CoreEvaluate):
    """Reads question + excerpts, returns one score per token."""

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(ids)).squeeze(-1)
