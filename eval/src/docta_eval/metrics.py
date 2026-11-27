"""Scores of one model's predictions against the gold questions.

Abstention is the phase-0 milestone metric: its positive class is "the model
should abstain", i.e. the question has no answer in the sources.
"""

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

Key = tuple[str, str]  # (world_id, question_id)


@dataclass(frozen=True)
class Gold:
    world_id: str
    question_id: str
    primitive: str
    hops: int
    answer: str | None
    required_sources: frozenset[str]
    # World entities the question names: what a search must be about to count.
    subjects: frozenset[str] = frozenset()
    trap: str | None = None
    expected_exit: str | None = None
    # What the false sources about the question's fact claim, as answers read.
    false_answers: frozenset[str] = frozenset()
    # Real text only (docta_eval.real): what the question asks whatever its
    # answer, and whether its absence is natural, redacted or none.
    shape: str | None = None
    absence: str | None = None
    # Other values the texts state for it, each as right: the key keeps
    # disagreements (1749 and 1750) on purpose.
    also: frozenset[str] = frozenset()

    def accepts(self, answer: str | None) -> bool:
        if self.answer is None or answer is None:
            return False
        return normalize(answer) in {normalize(a) for a in (self.answer, *self.also)}

    @classmethod
    def from_record(
        cls, q: dict, names: Iterable[str] = (), false_answers: Iterable[str] = ()
    ) -> "Gold":
        return cls(
            q["world_id"],
            q["question_id"],
            q["primitive"],
            q["hops"],
            q["answer"] if q["answerable"] else None,
            frozenset(q["required_sources"]),
            frozenset(n for n in names if mentions(q["text"], n)),
            q.get("trap"),
            q.get("expected_exit"),
            frozenset(false_answers),
        )


def mentions(text: str, name: str) -> bool:
    """Whether `text` names `name` as whole words ("Brask", not "Braskel")."""
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


@dataclass(frozen=True)
class Pred:
    abstained: bool
    answer: str | None
    cited_sources: frozenset[str]
    exit: str | None = None

    @classmethod
    def from_record(cls, p: dict) -> "Pred":
        return cls(p["abstained"], p["answer"], frozenset(p["cited_sources"]), p.get("exit"))


# A missing prediction is a wrong answer, never an abstention: otherwise a
# model that skips questions would earn abstention credit.
MISSING = Pred(abstained=False, answer=None, cited_sources=frozenset())


def normalize(text: str) -> str:
    """Exact match up to case, surrounding/repeated spaces and a final period."""
    return re.sub(r"\s+", " ", text).strip().removesuffix(".").strip().lower()


@dataclass
class Counts:
    questions: int = 0
    missing: int = 0
    answerable: int = 0
    correct: int = 0
    justified: int = 0
    cited: int = 0
    cited_right: int = 0
    required: int = 0
    abstain_tp: int = 0
    abstain_fp: int = 0
    abstain_fn: int = 0

    def add(self, gold: Gold, pred: Pred | None) -> None:
        self.questions += 1
        self.missing += pred is None
        p = pred or MISSING
        self.cited += len(p.cited_sources)
        self.cited_right += len(p.cited_sources & gold.required_sources)
        self.required += len(gold.required_sources)
        if gold.answer is None:
            self.abstain_tp += p.abstained
            self.abstain_fn += not p.abstained
            return
        self.answerable += 1
        self.abstain_fp += p.abstained
        right = not p.abstained and gold.accepts(p.answer)
        self.correct += right
        # Justified: the answer comes from evidence that establishes it, not
        # from knowing it. Extra citations are scored by citation precision.
        self.justified += right and gold.required_sources <= p.cited_sources


def ratio(num: int, den: int) -> float | None:
    """None when undefined, so 'no data' never reads as a score of 0."""
    return num / den if den else None


def summary(c: Counts) -> dict:
    tp, fp, fn = c.abstain_tp, c.abstain_fp, c.abstain_fn
    # Without any unanswerable question the positive class is empty: precision
    # and F1 would read one wrong abstention as total failure. The false
    # abstention rate says what happened instead.
    has_positives = tp + fn > 0
    return {
        "questions": c.questions,
        "missing": c.missing,
        "accuracy": ratio(c.correct, c.answerable),
        "justified_accuracy": ratio(c.justified, c.answerable),
        "citation_precision": ratio(c.cited_right, c.cited),
        "citation_recall": ratio(c.cited_right, c.required),
        "abstention_precision": ratio(tp, tp + fp) if has_positives else None,
        "abstention_recall": ratio(tp, tp + fn),
        "abstention_f1": ratio(2 * tp, 2 * tp + fp + fn) if has_positives else None,
        "false_abstention_rate": ratio(fp, c.answerable),
    }


def score(golds: list[Gold], preds: dict[Key, Pred]) -> dict:
    """Overall scores, then the same broken down by primitive and by hops,
    and on real text by shape and by kind of absence."""
    out = {
        "overall": _grouped(golds, preds, lambda g: "all").get("all", summary(Counts())),
        "by_primitive": _grouped(golds, preds, lambda g: g.primitive),
        "by_hops": _grouped(golds, preds, lambda g: str(g.hops)),
    }
    if any(g.shape for g in golds):
        out["by_shape"] = _grouped(golds, preds, lambda g: g.shape or "")
        out["by_absence"] = _grouped(golds, preds, lambda g: g.absence or "")
        _blank_fact_citations(out)
    return out


# A real paragraph carries no fact ids: these would read every citation as
# wrong and every right answer as justified. "supported" stands in for them.
FACT_CITATIONS = ("justified_accuracy", "citation_precision", "citation_recall")


def _blank_fact_citations(scores: dict) -> None:
    groups = [v for k, v in scores.items() if k != "overall"]
    for one in [scores["overall"], *(s for g in groups for s in g.values())]:
        one.update(dict.fromkeys(FACT_CITATIONS))


def _grouped(golds: list[Gold], preds: dict[Key, Pred], group: Callable[[Gold], str]) -> dict:
    counts: dict[str, Counts] = {}
    for g in golds:
        counts.setdefault(group(g), Counts()).add(g, preds.get((g.world_id, g.question_id)))
    return {name: summary(c) for name, c in sorted(counts.items())}
