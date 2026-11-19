"""The Evaluate core: a tiny byte-level transformer trained from scratch.

Bytes, not words: invented names never fall out of vocabulary, there is no
tokenizer to train, and nothing pretrained enters the model. Each byte also
carries a name mark (see marks.py), computed, so the core need not learn to
match spellings before it can judge what it read.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from docta_orchestrator.search import Hit
from torch import nn

from docta_train.marks import MAX_NAMES, name_marks

PAD, CLS, SEP = 0, 1, 2
OFFSET = 3  # byte b is token b + OFFSET
VOCAB = 256 + OFFSET


@dataclass(frozen=True)
class Config:
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 256
    max_len: int = 1024  # longest example is ~970 bytes: nothing truncated
    # Any dropout disables PyTorch's memory-efficient CPU attention, which then
    # keeps each length x length matrix for backward (~16 GB at batch 64).
    dropout: float = 0.0


class Encoded(NamedTuple):
    ids: list[int]  # CLS, then UTF-8 bytes shifted by OFFSET
    marks: list[int]  # one name mark per id, 0 for CLS and non-names


def encode(question: str, excerpts: Iterable[Hit], max_len: int) -> Encoded:
    """CLS, then the question and each excerpt as UTF-8 bytes, cut to max_len."""
    text = f"Q: {question}\n" + "".join(f"[{h.title}] {h.body}\n" for h in excerpts)
    ids = [CLS, *(b + OFFSET for b in text.encode("utf-8"))]
    marks = [0, *name_marks(question, text)]
    return Encoded(ids[:max_len], marks[:max_len])


def pad_batch(batch: Sequence[Encoded]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ids and marks padded to the longest sequence, and the mask of padding."""
    width = max(len(e.ids) for e in batch)
    ids = torch.full((len(batch), width), PAD, dtype=torch.long)
    marks = torch.zeros((len(batch), width), dtype=torch.long)
    for row, e in enumerate(batch):
        ids[row, : len(e.ids)] = torch.tensor(e.ids, dtype=torch.long)
        marks[row, : len(e.marks)] = torch.tensor(e.marks, dtype=torch.long)
    return ids, marks, ids == PAD


class CoreEvaluate(nn.Module):
    """Reads question + excerpts, returns the logit of "this is enough"."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(VOCAB, config.d_model, padding_idx=PAD)
        self.positions = nn.Embedding(config.max_len, config.d_model)
        self.marks = nn.Embedding(MAX_NAMES + 1, config.d_model)
        layer = nn.TransformerEncoderLayer(
            config.d_model,
            config.heads,
            config.ff,
            config.dropout,
            batch_first=True,
            norm_first=True,
        )
        # No nested tensors: they change float results with the amount of padding.
        self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, 1)

    def forward(
        self, ids: torch.Tensor, marks: torch.Tensor, pad_mask: torch.Tensor
    ) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions) + self.marks(marks)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(self.norm(x[:, 0])).squeeze(-1)
