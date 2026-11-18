"""Fake models and a hand-written rule: floors a trained core has to beat."""

from docta_orchestrator.models.abstain import AlwaysAbstainModel
from docta_orchestrator.models.keyword import KeywordModel
from docta_orchestrator.models.random_model import RandomModel
from docta_orchestrator.models.rule_evaluate import RuleEvaluateModel

__all__ = ["AlwaysAbstainModel", "KeywordModel", "RandomModel", "RuleEvaluateModel"]
