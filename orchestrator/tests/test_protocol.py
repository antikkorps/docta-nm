"""The human-correction protocol: each answerable chain run three times, with
no hint, the right bridge and a wrong one."""

import json

from docta_orchestrator.models import DigModel
from docta_orchestrator.protocol import hints, run_protocol
from docta_orchestrator.search import Hit

WORLD = {
    "world_id": "w1",
    "entities": [
        {"id": "e1", "kind": "person", "name": "Orlan Teste"},
        {"id": "e2", "kind": "city", "name": "Varnelle"},
        {"id": "e3", "kind": "city", "name": "Brask"},
    ],
    "facts": [
        {"id": "f1", "subj": "e1", "pred": "lives_in", "obj": "e2"},
        {"id": "f2", "subj": "e2", "pred": "founded_in", "obj": 1432},
        {"id": "f3", "subj": "e3", "pred": "founded_in", "obj": 1610},
    ],
}
CHAIN = {
    "world_id": "w1", "question_id": "q1", "primitive": "chain", "answerable": True,
    "text": "In what year was the city where Orlan Teste lives founded?",
    "required_facts": ["f1", "f2"],
}  # fmt: skip
INDEX = {
    "Varnelle": [Hit("s2", "Annals", "Varnelle was founded in 1432.", 0.9),
                 Hit("s1", "Notes", "Orlan Teste lives in Varnelle.", 0.9)],
    "Brask": [Hit("s3", "Annals", "Brask was founded in 1610.", 0.9)],
}  # fmt: skip


def search(query):
    return next((hits for name, hits in INDEX.items() if name in query), [])


def test_a_chain_gets_its_bridge_and_a_wrong_city_with_what_it_would_answer():
    h = hints(CHAIN, WORLD)
    assert h.right.body == "Orlan Teste lives in Varnelle."
    assert h.wrong.body == "Orlan Teste lives in Brask."
    assert (h.right_bridge, h.wrong_bridge, h.wrong_answer) == ("Varnelle", "Brask", "1610")


def test_only_answerable_chains_take_part():
    assert hints({**CHAIN, "primitive": "retrieve"}, WORLD) is None
    assert hints({**CHAIN, "answerable": False}, WORLD) is None


def test_each_model_runs_each_chain_without_a_hint_with_the_right_one_and_a_wrong_one(tmp_path):
    run_protocol([CHAIN], {"w1": WORLD}, [DigModel()], lambda wid: search, tmp_path)
    answers = {}
    for variant in ("none", "right", "wrong"):
        line = (tmp_path / f"dig.{variant}.jsonl").read_text().splitlines()[0]
        answers[variant] = json.loads(line)["answer"]
        assert (tmp_path / "traces" / f"dig.{variant}.jsonl").exists()
    assert answers == {"none": None, "right": "1432", "wrong": None}
    told = json.loads((tmp_path / "hints.jsonl").read_text().splitlines()[0])
    assert told == {"world_id": "w1", "question_id": "q1", "right": "Varnelle",
                    "wrong": "Brask", "wrong_answer": "1610"}  # fmt: skip
