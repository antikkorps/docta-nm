"""Reading gold questions and a model's predictions."""

import json
from pathlib import Path

from docta_orchestrator.jsonl import read_records

from docta_eval.metrics import Gold, Key, Pred

PREDICTION_FIELDS = ["question_id", "world_id", "abstained", "answer", "cited_sources"]


def read_gold(path: Path) -> list[Gold]:
    """The questions of a generated split; worlds and sources are skipped."""
    golds: dict[Key, Gold] = {}
    for where, record in read_records(path):
        if record["type"] != "question":
            continue
        key = (record["world_id"], record["question_id"])
        if key in golds:
            raise ValueError(f"{where}: question {'/'.join(key)} twice")
        answer = record["answer"]
        if record["answerable"] and not (isinstance(answer, str) and answer):
            raise ValueError(f"{where}: question is answerable but has no answer")
        golds[key] = Gold.from_record(record)
    if not golds:
        raise ValueError(f"{path}: no question to score")
    return list(golds.values())


def read_predictions(path: Path, gold_keys: set[Key]) -> dict[Key, Pred]:
    """One model's predictions; errors name the file and line to fix."""
    preds: dict[Key, Pred] = {}
    with Path(path).open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            where = f"{path}:{number}"
            record = _parse(line, where)
            key = (record["world_id"], record["question_id"])
            label = f"{key[0]}/{key[1]}"
            if key not in gold_keys:
                raise ValueError(f"{where}: {label} is not a gold question")
            if key in preds:
                raise ValueError(f"{where}: duplicate prediction for {label}")
            preds[key] = Pred.from_record(record)
    return preds


def _parse(line: str, where: str) -> dict:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as e:
        raise ValueError(f"{where}: invalid JSON: {e}") from e
    if not isinstance(record, dict):
        raise ValueError(f"{where}: expected a JSON object")
    missing = [k for k in PREDICTION_FIELDS if k not in record]
    if missing:
        raise ValueError(f"{where}: prediction lacks {', '.join(missing)}")
    problem = _type_problem(record)
    if problem:
        raise ValueError(f"{where}: {problem}")
    return record


def _type_problem(r: dict) -> str | None:
    """Rejects values that would otherwise be coerced into a silently wrong
    score: "false" read as true, "s1" read as the citations {"s", "1"}..."""
    for key in ("world_id", "question_id"):
        if not isinstance(r[key], str):
            return f"{key} must be a string"
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
