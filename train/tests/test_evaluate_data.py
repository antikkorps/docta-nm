"""The Evaluate dataset: what the core reads, and whether it is enough."""

from docta_orchestrator.search import Hit
from docta_train.evaluate_data import Example, build_examples, enough, read_examples, write_examples

S1 = Hit("s1", "Chronicle of Varnelle", "Varnelle was founded in 1432.", 0.9)
S2 = Hit("s2", "Notes on Orlan Teste", "Orlan Teste lives in Varnelle.", 0.7)


def question(qid, text, answerable, required):
    return {
        "type": "question", "world_id": "w1", "question_id": qid, "text": text,
        "answerable": answerable, "answer": "x" if answerable else None,
        "required_sources": required, "primitive": "retrieve", "hops": 1,
    }  # fmt: skip


def test_enough_only_when_every_needed_source_was_read():
    chain = question("q1", "?", True, ["s2", "s1"])
    assert enough(chain, [S1, S2])
    assert not enough(chain, [S1])  # the first hop is missing
    assert not enough(question("q2", "?", False, []), [S1, S2])  # nothing could be enough


def test_examples_hold_what_the_core_reads_from_a_real_search():
    asked = []

    def search(query):
        asked.append(query)
        return [S1]

    questions = [question("q1", "In what year was Varnelle founded?", True, ["s1"])]
    [example] = build_examples(questions, lambda world_id: search, "train")
    assert asked == ["In what year was Varnelle founded?"]
    assert example == Example(
        "w1", "q1", "train", "In what year was Varnelle founded?", (S1,), True
    )


def test_examples_round_trip_through_jsonl(tmp_path):
    examples = [Example("w1", "q1", "test", "Q?", (S1, S2), False)]
    path = tmp_path / "evaluate.jsonl"
    write_examples(examples, path)
    assert list(read_examples(path)) == examples
