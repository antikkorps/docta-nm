"""The Evaluate core: a tiny byte-level transformer trained from scratch.

Bytes, not words: invented names never fall out of vocabulary, there is no
tokenizer to train, and nothing pretrained enters the model.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from docta_orchestrator.search import Hit
from torch import nn

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
    dropout: float = 0.1


def encode(question: str, excerpts: Iterable[Hit], max_len: int) -> list[int]:
    """CLS, then the question and each excerpt as UTF-8 bytes, cut to max_len."""
    text = f"Q: {question}\n" + "".join(f"[{h.title}] {h.body}\n" for h in excerpts)
    return [CLS, *(b + OFFSET for b in text.encode("utf-8"))][:max_len]


def pad_batch(sequences: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids padded to the longest sequence, and the mask of padding."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), PAD, dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids, ids == PAD


class CoreEvaluate(nn.Module):
    """Reads question + excerpts, returns the logit of "this is enough"."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(VOCAB, config.d_model, padding_idx=PAD)
        self.positions = nn.Embedding(config.max_len, config.d_model)
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

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        x = self.encoder(x, src_key_padding_mask=pad_mask)
        return self.head(self.norm(x[:, 0])).squeeze(-1)
