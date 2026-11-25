"""The Evaluate core: a tiny transformer over word tokens, trained from scratch.

Tokens come from words.py: the templates' words, learned from the training
split, and the question's names as their rank. Nothing pretrained enters it.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn

from docta_train.words import BODY, PAD, SRC, STOP


@dataclass(frozen=True)
class Config:
    d_model: int = 128
    layers: int = 4
    heads: int = 4
    ff: int = 256
    # Longest example on 20,000 worlds of the richer language: 530 tokens
    # (train), 517 (ood); 320 cut 13% of them. Training refuses a longer one.
    max_len: int = 576
    # Each attention matrix (length x length per head) is kept for backward:
    # the sentence mask, like any dropout, rules out the memory-efficient kernel.
    dropout: float = 0.0
    local_heads: int = 2  # heads that read within one sentence; the others read all

    def __post_init__(self):
        # With every head local, <cls> never sees an excerpt: training would run
        # fine and learn nothing.
        if not 0 <= self.local_heads < self.heads:
            raise ValueError(f"local_heads must be in [0, heads), got {self.local_heads}")


def whole(seqs: Sequence[list[int]], max_len: int) -> Sequence[list[int]]:
    """A cut excerpt can lose the sentence a label or an answer rests on, so
    an example the core cannot read whole is refused, never cut."""
    over = [len(s) for s in seqs if len(s) > max_len]
    if over:
        raise ValueError(
            f"{len(over)} of {len(seqs)} examples are longer than max_len {max_len} "
            f"(longest {max(over)} tokens): raise Config.max_len and retrain"
        )
    return seqs


def pad_batch(batch: Sequence[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Token ids padded to the longest sequence, and the mask of padding."""
    width = max(len(seq) for seq in batch)
    ids = torch.full((len(batch), width), PAD, dtype=torch.long)
    for row, seq in enumerate(batch):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return ids, ids == PAD


def sentence_ids(ids: torch.Tensor) -> torch.Tensor:
    """Which sentence each token is in: 0 for the question, then one number per
    title and per body sentence, -1 for padding."""
    starts = (ids == SRC) | (ids == BODY)
    starts[:, 1:] |= ids[:, :-1] == STOP
    return starts.cumsum(1).masked_fill(ids == PAD, -1)


def attention_mask(ids: torch.Tensor, heads: int, local: int) -> torch.Tensor:
    """What each head may not attend to, (batch * heads, length, length).

    The core failed to tie a property to the name it is stated for (74% on
    balanced val): it saw "population" and the name somewhere, not in one
    sentence. Telling which words share a sentence is counting, so code does
    it: the first `local` heads read within their sentence only. Each position
    sees itself, so a padding row is never empty (an empty row gives NaN)."""
    sentence = sentence_ids(ids)
    same = sentence[:, :, None] == sentence[:, None, :]
    real = (sentence >= 0)[:, None, :].expand_as(same)
    itself = torch.eye(ids.shape[1], dtype=torch.bool, device=ids.device)
    allowed = torch.stack([same if h < local else real for h in range(heads)], 1) | itself
    return ~allowed.flatten(0, 1)


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

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """Each token read in context, normalised: (batch, length, d_model)."""
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.tokens(ids) + self.positions(positions)
        # The mask covers padding too.
        mask = attention_mask(ids, self.config.heads, self.config.local_heads)
        return self.norm(self.encoder(x, mask=mask))

    def forward(self, ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # `pad_mask` is ids == PAD, already in the attention mask; it stays in
        # the signature so callers pass pad_batch's pair unchanged.
        return self.head(self.encode(ids)[:, 0]).squeeze(-1)
