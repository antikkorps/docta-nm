"""Training steps and validation for the Evaluate core: deterministic, CPU-only."""

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from docta_train.model import Encoded, pad_batch

Batch = tuple[Sequence[Encoded], torch.Tensor]  # (encoded examples, 0/1 labels)
LOSS = nn.BCEWithLogitsLoss()


def run_epoch(
    model: nn.Module, optimiser: torch.optim.Optimizer, batches: Iterable[Batch]
) -> float:
    """One optimiser step per batch; returns the mean loss. The optimiser is
    passed in so AdamW's moments carry over from one epoch to the next."""
    model.train()
    losses = []
    for sequences, labels in batches:
        loss = LOSS(model(*pad_batch(sequences)), labels.float())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        losses.append(loss.item())
    return sum(losses) / len(losses)


def train_steps(model: nn.Module, batches: Iterable[Batch], lr: float) -> float:
    """A fresh AdamW run over `batches`, for tests and quick experiments."""
    return run_epoch(model, torch.optim.AdamW(model.parameters(), lr=lr), batches)


@torch.no_grad()
def validate(model: nn.Module, batches: Iterable[Batch]) -> dict:
    """Loss, accuracy, and precision/recall of the "enough" class."""
    model.eval()
    total = n = right = tp = fp = fn = 0
    for sequences, labels in batches:
        logits = model(*pad_batch(sequences))
        total += LOSS(logits, labels.float()).item() * len(labels)
        said, truth = logits > 0, labels > 0.5
        n += len(labels)
        right += (said == truth).sum().item()
        tp += (said & truth).sum().item()
        fp += (said & ~truth).sum().item()
        fn += (~said & truth).sum().item()
    return {
        "val_loss": total / n,
        "val_accuracy": right / n,
        "val_precision": tp / (tp + fp) if tp + fp else None,
        "val_recall": tp / (tp + fn) if tp + fn else None,
    }
