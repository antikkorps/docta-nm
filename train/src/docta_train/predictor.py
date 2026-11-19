"""Loads a trained Evaluate core and answers: is what I read enough?"""

from collections.abc import Iterable
from pathlib import Path

import torch
from docta_orchestrator.search import Hit

from docta_train.model import Config, CoreEvaluate, encode, pad_batch


class Predictor:
    def __init__(self, checkpoint: Path):
        saved = torch.load(checkpoint, weights_only=True)
        self.config = Config(**saved["config"])
        self.model = CoreEvaluate(self.config)
        try:
            self.model.load_state_dict(saved["state"])
        except RuntimeError as e:
            # Raised as ValueError so run_core reports it instead of a traceback.
            raise ValueError(
                f"{checkpoint} does not match the current core (trained before name "
                "marks?): rerun `just fit`"
            ) from e
        self.model.eval()

    @torch.no_grad()
    def prob_enough(self, question: str, excerpts: Iterable[Hit]) -> float:
        batch = pad_batch([encode(question, excerpts, self.config.max_len)])
        return torch.sigmoid(self.model(*batch)).item()
