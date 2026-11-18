"""A tiny v0 world, shaped like the brief's example, shared by the tests."""

WORLD = {
    "schema": "v0",
    "type": "world",
    "world_id": "w_test01",
    "seed": 123,
    "split": "test",
    "entities": [
        {"id": "e1", "kind": "city", "name": "Varnelle"},
        {"id": "e2", "kind": "person", "name": "Orlan Teste"},
    ],
    "facts": [
        {"id": "f1", "subj": "e1", "pred": "founded_in", "obj": 1432},
        {"id": "f2", "subj": "e2", "pred": "lives_in", "obj": "e1"},
    ],
}
SOURCES = [
    {
        "schema": "v0",
        "type": "source",
        "world_id": "w_test01",
        "source_id": "s1",
        "title": "Chronicle of Varnelle",
        "reliability": 0.9,
        "text": "Varnelle was founded in 1432.",
        "facts": ["f1"],
    },
    {
        "schema": "v0",
        "type": "source",
        "world_id": "w_test01",
        "source_id": "s2",
        "title": "Notes on Orlan Teste",
        "reliability": 0.7,
        "text": "Orlan Teste lives in Varnelle.",
        "facts": ["f2"],
    },
]
QUESTIONS = [
    {
        "schema": "v0",
        "type": "question",
        "world_id": "w_test01",
        "question_id": "q1",
        "text": "In what year was the city where Orlan Teste lives founded?",
        "primitive": "chain",
        "hops": 2,
        "answerable": True,
        "answer": "1432",
        "required_facts": ["f2", "f1"],
        "required_sources": ["s2", "s1"],
    },
    {
        "schema": "v0",
        "type": "question",
        "world_id": "w_test01",
        "question_id": "q2",
        "text": "In what year was Orlan Teste born?",
        "primitive": "spot_missing",
        "hops": 1,
        "answerable": False,
        "answer": None,
        "missing": "no source mentions the birth year of Orlan Teste",
        "required_facts": [],
        "required_sources": [],
    },
]
RECORDS = [WORLD, *SOURCES, *QUESTIONS]
