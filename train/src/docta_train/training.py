"""Training steps and validation for the Evaluate core: deterministic, CPU-only."""

from collections.abc import Iterable, Sequence

import torch
from torch import nn

from docta_train.model import pad_batch

Batch = tuple[Sequence[list[int]], torch.Tensor]  # (token ids, 0/1 labels)
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
def judged(model: nn.Module, batches: Iterable[Batch]) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Mean loss, then what the core said and what was true, in batch order."""
    model.eval()
    total, said, truth = 0.0, [], []
    for sequences, labels in batches:
        logits = model(*pad_batch(sequences))
        total += LOSS(logits, labels.float()).item() * len(labels)
        said.append(logits > 0)
        truth.append(labels > 0.5)
    said, truth = torch.cat(said), torch.cat(truth)
    return total / len(truth), said, truth


def validate(
    model: nn.Module, batches: Iterable[Batch], shapes: Sequence[str] | None = None
) -> dict:
    """Loss, accuracy, and precision/recall of the "enough" class. With the
    question shape of each example, in batch order, accuracy per shape too: it
    shows which kind of reasoning fails, which one overall score hides."""
    loss, said, truth = judged(model, batches)
    tp, fp, fn = (said & truth).sum().item(), (said & ~truth).sum().item(), (~said & truth).sum()
    metrics = {
        "val_loss": loss,
        "val_accuracy": (said == truth).float().mean().item(),
        "val_precision": tp / (tp + fp) if tp + fp else None,
        "val_recall": tp / (tp + fn.item()) if tp + fn.item() else None,
    }
    if shapes is not None:
        metrics["val_by_shape"] = by_shape((said == truth).tolist(), shapes)
    return metrics


def by_shape(right: Sequence[bool], shapes: Sequence[str]) -> dict[str, float]:
    counts: dict[str, list[int]] = {}
    for ok, shape in zip(right, shapes, strict=True):
        hit = counts.setdefault(shape, [0, 0])
        hit[0] += ok
        hit[1] += 1
    return {shape: r / n for shape, (r, n) in sorted(counts.items())}
