"""The Read core: points at the answer in what was read; code copies it.

The core never writes an answer, so it never needs a fact or even a digit:
it scores each word of the excerpts' bodies, and the orchestrator copies the
number, or the whole name, found at the best one, citing its source. It is
the Evaluate core's encoder with one score per token instead of one per
example: same word tokens, same sentence-local heads.
"""

from pathlib import Path

import torch
from torch import nn

from docta_train.model import Config, CoreEvaluate, pad_batch
from docta_train.words import BODY, PAD, SRC, STOP, WORD, Vocab

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
    """Reads question + excerpts, returns one score per token, from one of two
    pointers on the same encoding: what answers the question, or the name to
    search next. Taught to one pointer, the two pulled against each other."""

    def __init__(self, config: Config, vocab_size: int):
        super().__init__(config, vocab_size)
        self.bridge = nn.Linear(config.d_model, 1)

    def forward(
        self, ids: torch.Tensor, pad_mask: torch.Tensor, bridge: torch.Tensor | None = None
    ) -> torch.Tensor:
        """`bridge` marks the rows read for the next name; the others, for the answer."""
        encoded = self.encode(ids)
        answer = self.head(encoded).squeeze(-1)
        if bridge is None:
            return answer
        return torch.where(bridge[:, None], self.bridge(encoded).squeeze(-1), answer)


class ReadPredictor:
    """Loads a trained Read core and answers: what does the reading say?"""

    def __init__(self, checkpoint: Path):
        saved = torch.load(checkpoint, weights_only=True)
        if saved.get("kind") != "read":
            raise ValueError(f"{checkpoint} is not a Read core: train one with `just fit-read`")
        self.config = Config(**saved["config"])
        self.vocab = Vocab(saved["vocab"])
        self.model = CoreRead(self.config, len(self.vocab.words))
        if "bridge.weight" not in saved["state"]:
            stale = "predates the bridge pointer: rerun `just read-data && just fit-read`"
            raise ValueError(f"{checkpoint} {stale}")
        self.model.load_state_dict(saved["state"])
        self.model.eval()

    def read(self, question: str, excerpts) -> tuple[str, str | None] | None:
        """The answer copied at the best body word, with its source."""
        return self.point(question, excerpts, bridge=False)

    def bridge(self, question: str, excerpts) -> tuple[str, str | None] | None:
        """The name to search next, copied where it is said, with its source."""
        return self.point(question, excerpts, bridge=True)

    @torch.no_grad()
    def point(self, question: str, excerpts, bridge: bool) -> tuple[str, str | None] | None:
        excerpts = list(excerpts)
        ids, pad = pad_batch([self.vocab.encode(question, excerpts)[: self.config.max_len]])
        allowed = candidates(ids)
        if not allowed.any():
            return None
        scores = self.model(ids, pad, torch.tensor([bridge]))
        at = scores.masked_fill(~allowed, float("-inf")).argmax(1).item()
        return extract(aligned(question, excerpts), at)
