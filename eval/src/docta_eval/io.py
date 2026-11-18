"""Reading gold questions, and a model's predictions and traces."""

import json
from collections.abc import Callable
from pathlib import Path

from docta_orchestrator.jsonl import read_records

from docta_eval.metrics import Gold, Key, Pred

PREDICTION_FIELDS = ["question_id", "world_id", "abstained", "answer", "cited_sources"]
TRACE_FIELDS = ["question_id", "world_id", "states", "decision"]


def read_gold(path: Path) -> list[Gold]:
    """The questions of a generated split; worlds and sources are skipped."""
    golds: dict[Key, Gold] = {}
    names: dict[str, list[str]] = {}
    for where, record in read_records(path):
        if record["type"] == "world":
            names[record["world_id"]] = [e["name"] for e in record["entities"]]
        if record["type"] != "question":
            continue
        key = (record["world_id"], record["question_id"])
        if key in golds:
            raise ValueError(f"{where}: question {'/'.join(key)} twice")
        answer = record["answer"]
        if record["answerable"] and not (isinstance(answer, str) and answer):
            raise ValueError(f"{where}: question is answerable but has no answer")
        golds[key] = Gold.from_record(record, names.get(record["world_id"], []))
    if not golds:
        raise ValueError(f"{path}: no question to score")
    return list(golds.values())


def read_predictions(path: Path, gold_keys: set[Key]) -> dict[Key, Pred]:
    """One model's predictions; errors name the file and line to fix."""
    records = _read_keyed(path, gold_keys, PREDICTION_FIELDS, _type_problem, "prediction")
    return {key: Pred.from_record(r) for key, r in records.items()}


def read_traces(path: Path, gold_keys: set[Key]) -> dict[Key, dict]:
    """One model's traces, as `docta_orchestrator.baseline` writes them."""
    return _read_keyed(path, gold_keys, TRACE_FIELDS, _trace_problem, "trace")


def _read_keyed(
    path: Path,
    gold_keys: set[Key],
    fields: list[str],
    problem_of: Callable[[dict], str | None],
    what: str,
) -> dict[Key, dict]:
    """Per-question JSONL records, checked line by line against the gold set."""
    records: dict[Key, dict] = {}
    with Path(path).open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            where = f"{path}:{number}"
            record = _parse(line, where, fields, what)
            problem = problem_of(record)
            if problem:
                raise ValueError(f"{where}: {problem}")
            key = (record["world_id"], record["question_id"])
            label = f"{key[0]}/{key[1]}"
            if key not in gold_keys:
                raise ValueError(f"{where}: {label} is not a gold question")
            if key in records:
                raise ValueError(f"{where}: duplicate {what} for {label}")
            records[key] = record
    return records


def _parse(line: str, where: str, fields: list[str], what: str) -> dict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON: {e}") from e
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object")
    missing = [k for k in fields if k not in record]
    if missing:
        raise ValueError(f"{where}: {what} lacks {', '.join(missing)}")
    return record


def _ids_problem(r: dict) -> str | None:
    for key in ("world_id", "question_id"):
        if not isinstance(r[key], str):
            return f"{key} must be a string"
    return None


def _type_problem(r: dict) -> str | None:
    """Rejects values that would otherwise be coerced into a silently wrong
    score: "false" read as true, "s1" read as the citations {"s", "1"}..."""
    if problem := _ids_problem(r):
        return problem
    if not isinstance(r["abstained"], bool):
        return "abstained must be true or false"
    if r["answer"] is not None and not isinstance(r["answer"], str):
        return "answer must be a string or null"
    cited = r["cited_sources"]
    if not isinstance(cited, list) or not all(isinstance(c, str) for c in cited):
        return "cited_sources must be a list of strings"
    if r["abstained"] and r["answer"] is not None:
        return "prediction abstained but also answers"
    return None


STATES = {"frame", "plan", "consult", "evaluate", "confront", "conclude"}


def _trace_problem(r: dict) -> str | None:
    """Checks a trace has the shape the orchestrator writes, so scoring it can
    neither crash nor silently drop part of it."""
    if problem := _ids_problem(r):
        return problem
    states, decision = r["states"], r["decision"]
    if not isinstance(states, list) or not states or not isinstance(states[0], dict):
        return "states must start with a frame event"
    if states[0].get("state") != "frame":
        return "states must start with a frame event"
    if problem := _events_problem(states):
        return problem
    if not isinstance(decision, dict) or not isinstance(decision.get("abstained"), bool):
        return "decision.abstained must be true or false"
    concluded = any(e["state"] == "conclude" for e in states)
    if decision["abstained"] == concluded:
        return "decision disagrees with its states (abstained vs conclude)"
    return None


def _events_problem(states: list) -> str | None:
    planned = False
    for number, event in enumerate(states, start=1):
        if not isinstance(event, dict) or event.get("state") not in STATES:
            return f"event {number} has no known state"
        planned = planned or event["state"] == "plan"
        if event["state"] != "consult":
            continue
        if not planned:
            return f"event {number}: consult before any plan"
        excerpts = event.get("excerpts")
        if not isinstance(excerpts, list) or not all(
            isinstance(x, dict) and isinstance(x.get("source_id"), str) for x in excerpts
        ):
            return f"event {number}: excerpt without source_id"
    return None
