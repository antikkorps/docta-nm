"""The orchestrator: an explicit state machine that runs one question.

Each state is a small function taking the work so far and returning the next
state. The model fills states; transitions, searching, the round limit and
the trace belong to the orchestrator.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, replace

from docta_orchestrator.human import HINT_ROUND
from docta_orchestrator.search import Hit
from docta_orchestrator.states import (
    Confrontation,
    Evidence,
    Exit,
    Framing,
    Model,
    State,
)

Search = Callable[[str], list[Hit]]


@dataclass(frozen=True)
class Outcome:
    answer: str | None
    cited: tuple[str, ...]
    confidence: float | None
    exit: Exit | None


@dataclass(frozen=True)
class Work:
    question: str
    framing: Framing | None = None
    queries: tuple[str, ...] = ()
    evidence: Evidence = ()
    round_: int = 0
    confrontation: Confrontation | None = None
    outcome: Outcome | None = None


@dataclass
class Env:
    model: Model
    search: Search
    max_rounds: int
    hint: Hit | None = None  # what a human says from round HINT_ROUND on
    trace: list[dict] = field(default_factory=list)

    def log(self, state: State, **details) -> None:
        self.trace.append({"state": state.value, **details})


def run(
    question: str, model: Model, search: Search, max_rounds: int = 3, hint: Hit | None = None
) -> tuple[Outcome, list[dict]]:
    """Runs one question through the loop; returns its outcome and trace."""
    if max_rounds < 1:
        raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")
    env = Env(model, search, max_rounds, hint)
    state, work = State.FRAME, Work(question)
    while state is not State.DONE:
        state, work = STEPS[state](work, env)
    assert work.outcome is not None
    return work.outcome, env.trace


def _exit(work: Work, why: Exit) -> tuple[State, Work]:
    return State.DONE, replace(work, outcome=Outcome(None, (), None, why))


def _frame(work: Work, env: Env) -> tuple[State, Work]:
    f = env.model.frame(work.question)
    env.log(State.FRAME, question=f.question, unknowns=list(f.unknowns), exit=f.exit)
    work = replace(work, framing=f)
    return _exit(work, f.exit) if f.exit else (State.PLAN, work)


def _plan(work: Work, env: Env) -> tuple[State, Work]:
    queries = tuple(env.model.plan(work.framing, _heard(work, env), work.round_))
    env.log(State.PLAN, round=work.round_ + 1, queries=list(queries))
    work = replace(work, queries=queries, round_=work.round_ + 1)
    return (State.CONSULT if queries else State.EVALUATE), work


def _heard(work: Work, env: Env) -> Evidence:
    """What Plan reads: the evidence, and the human's hint once it is said.
    Only Plan: a hint may steer a search, never support an answer."""
    if env.hint is None or work.round_ < HINT_ROUND:
        return work.evidence
    if work.round_ == HINT_ROUND:
        env.trace.append({"state": "human", "round": work.round_ + 1, "hint": env.hint.body})
    return work.evidence + (env.hint,)


def _hint_pending(work: Work, env: Env) -> bool:
    """A human steps in when the model would give up before hearing them."""
    return env.hint is not None and work.round_ == HINT_ROUND < env.max_rounds


def _consult(work: Work, env: Env) -> tuple[State, Work]:
    seen = {h.source_id for h in work.evidence}
    new: list[Hit] = []
    for query in work.queries:
        for hit in env.search(query):
            if hit.source_id not in seen:
                seen.add(hit.source_id)
                new.append(hit)
    excerpts = [{"source_id": h.source_id, "title": h.title, "body": h.body} for h in new]
    env.log(State.CONSULT, queries=list(work.queries), excerpts=excerpts)
    return State.EVALUATE, replace(work, evidence=work.evidence + tuple(new))


def _evaluate(work: Work, env: Env) -> tuple[State, Work]:
    v = env.model.evaluate(work.framing, work.evidence)
    # "Enough" without a single excerpt is not enough: the method never states
    # anything it has not read.
    enough = v.enough and bool(work.evidence)
    out_of_rounds = not enough and work.round_ >= env.max_rounds
    why = v.exit or (Exit.NOT_FOUND if out_of_rounds else None)
    env.log(State.EVALUATE, enough=enough, exit=why)
    if why is Exit.NOT_FOUND and _hint_pending(work, env):
        return State.PLAN, work
    if why:
        return _exit(work, why)
    return (State.CONFRONT if enough else State.PLAN), work


def _confront(work: Work, env: Env) -> tuple[State, Work]:
    c = env.model.confront(work.framing, work.evidence)
    env.log(State.CONFRONT, agreements=list(c.agreements), exit=c.exit)
    work = replace(work, confrontation=c)
    return _exit(work, c.exit) if c.exit else (State.CONCLUDE, work)


def _conclude(work: Work, env: Env) -> tuple[State, Work]:
    c = env.model.conclude(work.framing, work.evidence, work.confrontation)
    consulted = {h.source_id for h in work.evidence}
    # Traceability: a model may only cite what it actually read.
    cited = tuple(s for s in c.cited if s in consulted)
    dropped = [s for s in c.cited if s not in consulted]
    env.log(
        State.CONCLUDE,
        answer=c.answer,
        cited=list(cited),
        confidence=c.confidence,
        dropped_citations=dropped,
    )
    return State.DONE, replace(work, outcome=Outcome(c.answer, cited, c.confidence, None))


STEPS: dict[State, Callable[[Work, Env], tuple[State, Work]]] = {
    State.FRAME: _frame,
    State.PLAN: _plan,
    State.CONSULT: _consult,
    State.EVALUATE: _evaluate,
    State.CONFRONT: _confront,
    State.CONCLUDE: _conclude,
}
