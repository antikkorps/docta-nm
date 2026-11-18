"""Five gold questions covering every primitive, and hand-written predictions."""


def question(qid, primitive, hops, answer, required, text=None):
    return {
        "schema": "v1",
        "type": "question",
        "world_id": "w1",
        "question_id": qid,
        "text": text or f"question {qid}?",
        "primitive": primitive,
        "hops": hops,
        "answerable": answer is not None,
        "answer": answer,
        "required_facts": [f"f_{s}" for s in required],
        "required_sources": required,
        "trap": None,
        "expected_exit": None,
        **({} if answer is not None else {"missing": "no source mentions it"}),
    }


GOLD = [
    question("q1", "retrieve", 1, "1432", ["s1"]),
    question("q2", "chain", 2, "Varnelle", ["s2", "s1"]),
    question("q3", "compare", 1, "Brask", ["s3", "s4"]),
    question("q4", "spot_missing", 1, None, [], "In what year was Brask founded?"),
    question("q5", "spot_missing", 2, None, [], "How tall is Orlan Teste, in centimetres?"),
]
# Entity names of world w1, as its world record would list them.
NAMES = {"Varnelle", "Brask", "Orlan Teste"}


def pred(qid, answer=None, cited=(), abstained=False):
    return {
        "question_id": qid,
        "world_id": "w1",
        "abstained": abstained,
        "answer": answer,
        "cited_sources": list(cited),
        "confidence": 0.5,
        "exit": "not_found" if abstained else None,
    }


# Right on q1 (despite spacing, case and a final period) and q2; abstains
# wrongly on q3 and rightly on q4; answers q5, which has no answer.
MIXED = [
    pred("q1", " 1432. ", ["s1"]),
    pred("q2", "varnelle", ["s2"]),
    pred("q3", abstained=True),
    pred("q4", abstained=True),
    pred("q5", "Brask", ["s9"]),
]
