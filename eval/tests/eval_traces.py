"""Hand-written traces for the five gold questions of eval_sample."""

# What each source says; unlisted ones mention no entity of the questions.
BODIES = {"s6": "Brask keeps its seat in Varnelle.", "s7": "The rolls list 300 members."}


def rounds(*reads):
    """One plan/consult/evaluate round per tuple of source ids read."""
    events = []
    for i, ids in enumerate(reads, start=1):
        events.append({"state": "plan", "round": i, "queries": ["q"] if ids else []})
        if ids:
            excerpts = [
                {"source_id": s, "title": s, "body": BODIES.get(s, "nothing here")} for s in ids
            ]
            events.append({"state": "consult", "queries": ["q"], "excerpts": excerpts})
        events.append({"state": "evaluate", "enough": i == len(reads), "exit": None})
    return events


def trace(qid, events, abstained):
    frame = {"state": "frame", "question": qid, "unknowns": [], "exit": None}
    end = [] if abstained else [{"state": "conclude", "answer": "x", "cited": []}]
    if abstained:
        events[-1] = {**events[-1], "enough": False, "exit": "not_found"}
    return {
        "world_id": "w1",
        "question_id": qid,
        "model": "m",
        "question": qid,
        "states": [frame, *events, *end],
        "decision": {"question_id": qid, "world_id": "w1", "abstained": abstained},
    }


TRACES = [
    trace("q1", rounds(["s1"]), False),  # needs s1: read it, concluded on time
    trace("q2", rounds(["s2"], ["s9"]), False),  # needs s2+s1: never read s1, concluded early
    trace("q3", rounds(["s3", "s4"], ["s5"]), False),  # needs s3+s4: had both, searched once more
    trace("q4", rounds(["s6"]), True),  # no answer: read about Brask, then abstained
    trace("q5", rounds([]), True),  # no answer: abstained without reading anything
]
