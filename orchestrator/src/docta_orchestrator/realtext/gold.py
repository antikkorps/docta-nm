"""What the loop may see of a prepared real-text domain: each question's
text and where to search, nothing of its answer, values or subjects."""

from pathlib import Path

from docta_orchestrator.realtext.load import world_of
from docta_orchestrator.realtext.probe import read


def world(entities: list[dict]) -> str:
    """The world the loader made: named from the domain the collector wrote."""
    if not entities:
        raise ValueError("no entities: is entities.jsonl from realtext.collect?")
    return world_of(entities[0]["domain"])


def asked(where: Path) -> list[dict]:
    wid = world(read(where / "entities.jsonl"))
    return [{"world_id": wid, "question_id": q["question_id"], "text": q["body"]}
            for q in read(where / "questions.jsonl")]  # fmt: skip
