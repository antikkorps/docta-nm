"""Scores on trap questions (traps came with JSONL v1): the challenge step of the method.

A trap surrounds a question's fact with sources that disagree. Answering is
not enough: the model must not be fooled by the false side, and when the
sources cannot be settled it must say so (exit unresolved_contradiction)
rather than pick one.
"""

from dataclasses import dataclass

from docta_eval.metrics import MISSING, Gold, Key, Pred, normalize, ratio

UNRESOLVED = "unresolved_contradiction"


@dataclass
class Counts:
    questions: int = 0
    missing: int = 0
    decidable: int = 0
    right: int = 0
    justified: int = 0
    fooled: int = 0
    exit_tp: int = 0
    exit_fp: int = 0
    exit_fn: int = 0

    def add(self, gold: Gold, pred: Pred | None) -> None:
        self.questions += 1
        self.missing += pred is None
        p = pred or MISSING
        exited, expected = p.exit == UNRESOLVED, gold.expected_exit == UNRESOLVED
        self.exit_tp += exited and expected
        self.exit_fp += exited and not expected
        self.exit_fn += expected and not exited
        said = None if p.abstained or p.answer is None else normalize(p.answer)
        if gold.answer is None:
            # Undecidable: any answer settles what the sources cannot.
            self.fooled += said is not None
            return
        self.decidable += 1
        right = said == normalize(gold.answer)
        self.right += right
        self.justified += right and gold.required_sources <= p.cited_sources
        # A false source may happen to state the truth's value: right is right.
        self.fooled += not right and said in {normalize(a) for a in gold.false_answers}


def summary(c: Counts) -> dict:
    tp, fp, fn = c.exit_tp, c.exit_fp, c.exit_fn
    return {
        "questions": c.questions,
        "missing": c.missing,
        "accuracy": ratio(c.right, c.decidable),
        "justified_accuracy": ratio(c.justified, c.decidable),
        "fooled_rate": ratio(c.fooled, c.questions),
        "exit_precision": ratio(tp, tp + fp),
        "exit_recall": ratio(tp, tp + fn),
    }


def trap_scores(golds: list[Gold], preds: dict[Key, Pred]) -> dict[str, dict]:
    """Scores per trap kind and over all trap questions; empty without traps.
    The "all" exit precision also counts unresolved exits on plain questions:
    a model that never answers would otherwise look precise."""
    if all(g.trap is None for g in golds):
        return {}
    counts: dict[str, Counts] = {"all": Counts()}
    for g in golds:
        p = preds.get((g.world_id, g.question_id))
        if g.trap is None:
            counts["all"].exit_fp += p is not None and p.exit == UNRESOLVED
            continue
        for group in (g.trap, "all"):
            counts.setdefault(group, Counts()).add(g, p)
    return {group: summary(c) for group, c in sorted(counts.items())}
