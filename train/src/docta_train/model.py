"""The Evaluate core: a tiny transformer over word tokens, trained from scratch.

Tokens come from words.py: the templates' words, learned from the training
split, and the question's names as their rank. Nothing pretrained enters it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from docta_train.words import PAD


@dataclass(frozen=True)
class Config:
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 256
    max_len: int = 256  # longest example is 157 words on 20,000 worlds
    # Any dropout disables PyTorch's memory-efficient CPU attention, which then
    # keeps each length x length matrix for backward.
    dropout: float = 0.0


def pad_batch(batch: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids padded to the longest sequence, and the mask of padding."""
    width = max(len(seq) for seq in batch)
    ids = torch.full((len(batch), width), PAD, dtype=torch.long)
    for row, seq in enumerate(batch):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids, ids == PAD


class CoreEvaluate(nn.Module):
    """Reads question + excerpts, returns the logit of "this is enough"."""

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config
        self.tokens = nn.Embedding(vocab_size, config.d_model, padding_idx=PAD)
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
