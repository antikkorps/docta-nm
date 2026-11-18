"""Scores of the reflection itself, read from the orchestrator's traces.

Two models with the same answers can reason very differently: one reads what
it needs and stops, the other guesses, wanders or gives up without looking.
These metrics make that difference visible.
"""

from dataclasses import dataclass

from docta_eval.metrics import Gold, Key, mentions, ratio


def reads_per_round(states: list[dict]) -> list[set[str]]:
    """Source ids read in each search round, in order. A plan with no query
    searches nothing, so it is not a round."""
    rounds: list[set[str]] = []
    for event in states:
        if event["state"] == "plan" and event.get("queries"):
            rounds.append(set())
        elif event["state"] == "consult" and rounds:
            rounds[-1].update(x["source_id"] for x in event["excerpts"])
    return rounds


def read_about(states: list[dict], subjects: frozenset[str]) -> bool:
    """Whether some excerpt read names an entity the question is about."""
    return any(
        mentions(f"{x.get('title', '')} {x.get('body', '')}", name)
        for e in states
        if e["state"] == "consult"
        for x in e["excerpts"]
        for name in subjects
    )


def timing(rounds: list[set[str]], required: frozenset[str]) -> str:
    """'early' if it concluded before reading every needed source, 'late' if it
    kept searching after having them all, else 'on_time'."""
    seen: set[str] = set()
    for number, read in enumerate(rounds, start=1):
        seen |= read
        if required <= seen:
            return "late" if number < len(rounds) else "on_time"
    return "early"


@dataclass
class Counts:
    traces: int = 0
    missing: int = 0
    rounds: int = 0
    required: int = 0
    read: int = 0
    read_useful: int = 0
    answerable: int = 0
    concluded: int = 0
    early: int = 0
    late: int = 0
    abstentions: int = 0
    searched_abstentions: int = 0
    unanswerable: int = 0
    grounded: int = 0

    def add(self, gold: Gold, trace: dict | None) -> None:
        if trace is None:
            self.missing += 1
            return
        self.traces += 1
        rounds = reads_per_round(trace["states"])
        read = set().union(*rounds)
        self.rounds += len(rounds)
        searched = bool(read)
        if trace["decision"]["abstained"]:
            self.abstentions += 1
            self.searched_abstentions += searched
        if gold.answer is None:
            self.unanswerable += 1
            on_subject = read_about(trace["states"], gold.subjects)
            self.grounded += trace["decision"]["abstained"] and on_subject
            return
        self.answerable += 1
        self.required += len(gold.required_sources)
        self.read += len(read)
        self.read_useful += len(read & gold.required_sources)
        if any(e["state"] == "conclude" for e in trace["states"]):
            self.concluded += 1
            when = timing(rounds, gold.required_sources)
            self.early += when == "early"
            self.late += when == "late"


def process_scores(golds: list[Gold], traces: dict[Key, dict]) -> dict:
    c = Counts()
    for g in golds:
        c.add(g, traces.get((g.world_id, g.question_id)))
    return {
        "traces": c.traces,
        "missing": c.missing,
        "reading_recall": ratio(c.read_useful, c.required),
        "reading_precision": ratio(c.read_useful, c.read),
        "mean_rounds": ratio(c.rounds, c.traces),
        "conclusion_rate": ratio(c.concluded, c.answerable),
        "early_conclusion_rate": ratio(c.early, c.concluded),
        "late_conclusion_rate": ratio(c.late, c.concluded),
        "searched_before_abstaining": ratio(c.searched_abstentions, c.abstentions),
        "grounded_abstention_rate": ratio(c.grounded, c.unanswerable),
    }
