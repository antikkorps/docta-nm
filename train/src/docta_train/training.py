"""Training steps and validation for the Evaluate core: deterministic, CPU-only."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from docta_train.model import pad_batch

Batch = tuple[Sequence[list[int]], torch.Tensor]  # (token ids, 0/1 labels)
LOSS = nn.BCEWithLogitsLoss()


@dataclass
class Warmup:
    """The learning rate of each step: a straight climb over the first `steps`
    ones, then `lr`. Without it a high rate locks a fresh transformer on one
    constant answer (seen at 1e-3: two epochs at 50%)."""

    lr: float
    steps: int
    done: int = 0  # steps taken so far, across epochs and resumes

    def next(self) -> float:
        self.done += 1
        return self.lr * min(1.0, self.done / self.steps) if self.steps else self.lr


class EmptySplit(ValueError):
    """A split with nothing to learn or measure from: no data, not a setting."""


def require(examples: Sequence, split: str) -> None:
    """An empty split would end its first epoch on a division by zero, after
    the minutes spent loading: refuse it at once, saying how to get data."""
    if not examples:
        raise EmptySplit(
            f"no {split} example: generate more worlds (`just gen 42 1000`) and rebuild "
            "the data; caps (--max-train, --max-val) and balancing can also empty a split"
        )


def run_epoch(
    model: nn.Module,
    optimiser: torch.optim.Optimizer,
    batches: Iterable[Batch],
    warmup: Warmup | None = None,
) -> float:
    """One optimiser step per batch; returns the mean loss. The optimiser is
    passed in so AdamW's moments carry over from one epoch to the next."""
    model.train()
    losses = []
    for sequences, labels in batches:
        if warmup is not None:
            for group in optimiser.param_groups:
                group["lr"] = warmup.next()
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
    tp, fp, fn = ((a & b).sum().item() for a, b in ((said, truth), (said, ~truth), (~said, truth)))
    metrics = {
        "val_loss": loss,
        "val_accuracy": (said == truth).sum().item() / len(truth),
        "val_precision": tp / (tp + fp) if tp + fp else None,
        "val_recall": tp / (tp + fn) if tp + fn else None,
    }
    if shapes is not None:
        metrics["val_by_shape"] = by_shape((said == truth).tolist(), shapes)
    return metrics


def by_shape(right: Sequence[bool], shapes: Sequence[str]) -> dict[str, float]:
    """Share of right answers per question shape, shapes in sorted order."""
    counts: dict[str, list[int]] = {}
    for ok, key in zip(right, shapes, strict=True):
        tally = counts.setdefault(key, [0, 0])
        tally[0] += ok
        tally[1] += 1
    return {key: r / n for key, (r, n) in sorted(counts.items())}
