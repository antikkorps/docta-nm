"""Phase-0 fake models: floors the first trained core has to beat."""

from docta_orchestrator.models.abstain import AlwaysAbstainModel
from docta_orchestrator.models.keyword import KeywordModel
from docta_orchestrator.models.random_model import RandomModel

__all__ = ["AlwaysAbstainModel", "KeywordModel", "RandomModel"]
