"""Reading the generator's JSONL v2 output, the canonical dataset format.

Every record is checked as it is read, so any later failure is impossible or
names the file and line to fix.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = "v2"

# Fields each record type must carry; `missing` is optional on questions.
REQUIRED = {
    "world": ["world_id", "seed", "split", "entities", "facts"],
    "source": [
        "world_id", "source_id", "title", "reliability", "text", "phrasings",
        "facts", "false_claims", "copy_of",
    ],
    "question": [
        "world_id", "question_id", "text", "template", "primitive", "hops", "answerable",
        "answer", "required_facts", "required_sources", "trap", "expected_exit",
    ],
}  # fmt: skip
NESTED = {"entities": ["id", "kind", "name"], "facts": ["id", "subj", "pred", "obj"]}
# Closed vocabularies of question fields (None = no trap, no exit expected).
ALLOWED = {
    "trap": {None, "contradiction", "false_consensus", "undecidable"},
    "expected_exit": {None, "unresolved_contradiction"},
}

Located = tuple[str, dict]


@dataclass
class WorldBatch:
    world: dict
    sources: list[dict] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    # "file:line" of the world record, for error messages only.
    where: str = field(default="", compare=False)


def read_records(path: Path) -> Iterator[Located]:
    """Yields ("file:line", record) for each line, after checking it."""
    with Path(path).open(encoding="utf-8") as lines:
        for number, line in enumerate(lines, start=1):
            where = f"{path}:{number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{where}: invalid JSON: {e}") from e
            problem = _problem(record)
            if problem:
                raise ValueError(f"{where}: {problem}")
            yield where, record


def _problem(record: object) -> str | None:
    """What is wrong with a parsed line, or None if it is a valid v2 record."""
    if not isinstance(record, dict):
        return f"expected a JSON object, got {type(record).__name__}"
    if record.get("schema") != SCHEMA:
        return f"schema {record.get('schema')!r}, expected {SCHEMA}"
    kind = record.get("type")
    if kind not in REQUIRED:
        return f"unknown record type {kind!r}"
    missing = [k for k in REQUIRED[kind] if k not in record]
    if kind == "question" and not missing:
        wrong = [f"{k} {record[k]!r}" for k, ok in ALLOWED.items() if record[k] not in ok]
        if wrong:
            return f"question has an unknown {', '.join(wrong)}"
    if kind == "world" and not missing:
        missing = [
            f"{name}[{i}].{k}"
            for name, keys in NESTED.items()
            for i, item in enumerate(record[name])
            for k in keys
            if k not in item
        ]
    return f"{kind} lacks {', '.join(missing)}" if missing else None


def group_worlds(records: Iterable[Located]) -> Iterator[WorldBatch]:
    """Groups each world with the sources and questions that follow it, as the
    generator writes them. Streaming keeps memory flat on large files."""
    batch: WorldBatch | None = None
    for where, record in records:
        if record["type"] == "world":
            if batch:
                yield batch
            batch = WorldBatch(record, where=where)
        elif batch is None or record["world_id"] != batch.world["world_id"]:
            raise ValueError(f"{where}: {record['type']} of {record['world_id']} before its world")
        elif record["type"] == "source":
            batch.sources.append(record)
        else:
            batch.questions.append(record)
    if batch:
        yield batch
