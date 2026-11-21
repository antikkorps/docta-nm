"""The Read dataset: what the core reads once it judged it enough, and where
the answer sits in it (ground truth, for training and scoring only)."""

from docta_orchestrator.search import Hit
from docta_train.evaluate_data import Example
from docta_train.read_data import ReadExample, build_read_examples, read_reads, write_reads

WORLD = {
    "type": "world", "world_id": "w1",
    "entities": [
        {"id": "e1", "kind": "city", "name": "Varnelle"},
        {"id": "e2", "kind": "city", "name": "Brask"},
    ],
    "facts": [
        {"id": "f1", "subj": "e1", "pred": "founded_in", "obj": 1432},
        {"id": "f2", "subj": "e2", "pred": "founded_in", "obj": 1610},
    ],
}  # fmt: skip
S1 = Hit("s1", "Annals", "Varnelle was founded in 1432.", 0.9)
S2 = Hit("s2", "Annals", "Brask was founded in 1610.", 0.9)


def question(qid, text, primitive, answer, facts, sources, trap=None):
    return {
        "type": "question", "world_id": "w1", "question_id": qid, "text": text,
        "primitive": primitive, "answerable": answer is not None, "answer": answer,
        "required_facts": facts, "required_sources": sources, "trap": trap,
    }  # fmt: skip


def read(q, label=True, hits=(S1, S2)):
    example = Example("w1", q["question_id"], "val", q["text"], hits, label)
    return list(build_read_examples([WORLD, q], [example]))


def test_a_retrieve_question_reads_its_answer_in_its_sources():
    q = question("q1", "In what year was Varnelle founded?", "retrieve", "1432", ["f1"], ["s1"])
    assert read(q) == [
        ReadExample("w1", "q1", "val", q["text"], (S1, S2), "1432", ("s1",)),
    ]


def test_a_comparison_reads_each_option_value_with_the_option_named_last():
    # Code compares the two values: the core only reads them.
    text = "Which was founded earlier, Varnelle or Brask?"
    q = question("q2", text, "compare", "Varnelle", ["f1", "f2"], ["s1", "s2"])
    assert [(e.question, e.answer) for e in read(q)] == [
        (f"{text} Varnelle", "1432"),
        (f"{text} Brask", "1610"),
    ]


def test_only_what_was_judged_enough_and_holds_no_trap_is_read():
    q = question("q3", "In what year was Varnelle founded?", "retrieve", "1432", ["f1"], ["s1"])
    assert read(q, label=False) == []
    assert read({**q, "trap": "contradiction"}) == []


def test_chains_are_read_hop_by_hop_so_none_is_kept_whole():
    text = "In what year was the city where Orlan Teste lives founded?"
    assert read(question("q4", text, "chain", "1432", ["f9", "f1"], ["s9", "s1"])) == []


def test_reads_round_trip(tmp_path):
    q = question("q1", "In what year was Varnelle founded?", "retrieve", "1432", ["f1"], ["s1"])
    examples = read(q)
    write_reads(examples, tmp_path / "val.jsonl")
    assert list(read_reads(tmp_path / "val.jsonl")) == examples
