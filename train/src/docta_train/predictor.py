"""Loads a trained Evaluate core and answers: is what I read enough?"""

from collections.abc import Iterable
from pathlib import Path

import torch
from docta_orchestrator.search import Hit

from docta_train.model import Config, CoreEvaluate, pad_batch
from docta_train.words import Vocab


class Predictor:
    def __init__(self, checkpoint: Path):
        saved = torch.load(checkpoint, weights_only=True)
        try:
            self.config = Config(**saved["config"])
            self.vocab = Vocab(saved["vocab"])
            self.model = CoreEvaluate(self.config, len(self.vocab.words))
            self.model.load_state_dict(saved["state"])
        except (KeyError, TypeError, RuntimeError) as e:
            # Raised as ValueError so run_core reports it instead of a traceback.
            raise ValueError(
                f"{checkpoint} does not match the current core (trained before word "
                "tokens?): rerun `just fit`"
            ) from e
        self.model.eval()

    @torch.no_grad()
    def prob_enough(self, question: str, excerpts: Iterable[Hit]) -> float:
        batch = pad_batch([self.vocab.encode(question, excerpts)[: self.config.max_len]])
        return torch.sigmoid(self.model(*batch)).item()
