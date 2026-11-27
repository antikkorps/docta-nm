"""A tiny collected domain, prepared by the probe as `just real` will."""

import json

from docta_orchestrator.realtext import probe


def city(q, label, years):
    return {"domain": "cities", "kind": "city", "qid": q, "label": label, "title": label,
            "facts": {"inception": years}}  # fmt: skip


def article(q, *paragraphs):
    return {"qid": q, "title": q, "sections": [{"path": [q], "paragraphs": list(paragraphs)}]}


ENTITIES = [city("Q1", "Varna", [1749]), city("Q2", "Brask", [1802]),
            city("Q3", "Tolm", [1901]), city("Q4", "Kesh", [1650])]  # fmt: skip
ARTICLES = [
    article("Q1", "Varna was established in 1749. It lies on a river."),
    article("Q2", "Brask is a town on a hill."),
    article("Q3", "Tolm was founded in 1901.", "Rain falls on Tolm."),
    article("Q4", "Kesh was founded in 1650."),
]


def domain(tmp_path):
    for name, rows in (("entities", ENTITIES), ("articles", ARTICLES)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert probe.main([str(tmp_path), "--seed", "0", "--share", "0.5"]) == 0
    return tmp_path


def asked(where):
    return [json.loads(line) for line in (where / "questions.jsonl").open()]
