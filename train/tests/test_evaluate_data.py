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


CHAIN = "In what year was the city where Orlan Teste lives founded?"


def dug(bridges, first=(S2,)):
    """Examples for a chain whose first search reads `first`, the bridge's
    search Varnelle's chronicle."""

    def search(query):
        return list(first) if query == CHAIN else [S2, S1]

    q = question("q1", CHAIN, True, ["s2", "s1"])
    return list(build_examples([q], lambda world_id: search, "train", bridges))


def test_a_chain_is_also_judged_after_its_bridge_was_searched():
    # The dig adds what the bridge's search brings, the first search's
    # sources once: judged as the core sees it in round 2.
    first, after = dug({("w1", "q1"): ("Varnelle", "s2")})
    assert (first.excerpts, first.label) == ((S2,), False)
    assert (after.question, after.excerpts, after.label) == (CHAIN, (S2, S1), True)


def test_without_a_known_bridge_the_name_a_sentence_about_the_subject_brings_is_dug():
    _, after = dug({})
    assert after.excerpts == (S2, S1)


def test_no_dig_without_a_name_to_search_or_once_enough():
    assert len(dug({("w1", "q1"): ("Varnelle", "s2")}, first=(S1,))) == 1  # nothing names it
    assert len(dug({("w1", "q1"): ("Varnelle", "s2")}, first=(S2, S1))) == 1  # enough already


def test_a_one_hop_question_is_judged_after_a_dig_too():
    # core_bridge digs whatever name the core points at: Evaluate must have
    # seen more read than needed for a question that has no answer.
    q = question("q2", "How tall is Orlan Teste?", False, [])

    def search(query):
        return [S2] if query == q["text"] else [S1]

    first, after = build_examples([q], lambda world_id: search, "train", {})
    assert (after.excerpts, after.label) == ((S2, S1), False)
