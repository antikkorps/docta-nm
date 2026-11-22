"""A chain's bridge from ground truth: a training target, never an input."""

from docta_train.chain_truth import chain_bridges

WORLD = {
    "type": "world", "world_id": "w1",
    "entities": [
        {"id": "e1", "kind": "city", "name": "Varnelle"},
        {"id": "e2", "kind": "guild", "name": "Kel Dorn"},
        {"id": "e3", "kind": "person", "name": "Orlan Teste"},
    ],
    "facts": [
        {"id": "f1", "subj": "e1", "pred": "founded_in", "obj": 1432},
        {"id": "f2", "subj": "e3", "pred": "lives_in", "obj": "e1"},
        {"id": "f3", "subj": "e3", "pred": "member_of", "obj": "e2"},
    ],
}  # fmt: skip
LIVES = {"type": "source", "world_id": "w1", "source_id": "s2", "facts": ["f2"]}
GUILD = {"type": "source", "world_id": "w1", "source_id": "s3", "facts": ["f3"]}


def question(qid, text, hops=2, answerable=True, facts=("f2", "f1"), sources=("s2", "s1"),
             trap=None):  # fmt: skip
    return {
        "type": "question", "world_id": "w1", "question_id": qid, "text": text,
        "hops": hops, "answerable": answerable, "required_facts": list(facts),
        "required_sources": list(sources), "trap": trap,
    }  # fmt: skip


def bridges(*questions, sources=(LIVES, GUILD)):
    return chain_bridges([WORLD, *sources, *questions])


def test_an_answerable_chain_bridges_through_its_first_fact():
    q = question("q1", "In what year was the city where Orlan Teste lives founded?")
    assert bridges(q) == {("w1", "q1"): ("Varnelle", "s2")}


def test_an_unanswerable_chain_bridges_through_the_subjects_name_of_the_asked_kind():
    # Its required facts are empty; the subject and the kind asked still say
    # which name the core should search next, if a source states the link.
    q = question("q2", "How many members does the guild Orlan Teste is a member of have?",
                 answerable=False, facts=(), sources=())  # fmt: skip
    assert bridges(q) == {("w1", "q2"): ("Kel Dorn", "s3")}
    assert bridges(q, sources=(LIVES,)) == {}  # no source states the link


def test_one_hop_questions_and_traps_have_no_bridge():
    assert bridges(question("q3", "In which city does Orlan Teste live?", hops=1)) == {}
    trap = question("q4", "In what year was the city where Orlan Teste lives founded?",
                    trap="contradiction")  # fmt: skip
    assert bridges(trap) == {}
