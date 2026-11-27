"""The real-text probe as gold (a folder `realtext.probe` prepared), and its
"supported" citations.

A paragraph has no fact ids, so a citation cannot be checked against the
facts an answer needs. Weaker, and named apart: a right answer is supported
when, for each entity whose year it rests on, a cited paragraph holds that
year and is the entity's own article or names it. Looser than the line
between answerable and not (stated.py), which wants name and year in one
sentence; and a chain's link ("born in") is not checked, only its year.
"""

from dataclasses import dataclass
from pathlib import Path

from docta_orchestrator.realtext.gold import world
from docta_orchestrator.realtext.load import sources
from docta_orchestrator.realtext.probe import read
from docta_orchestrator.realtext.questions import asked, named
from docta_orchestrator.realtext.redact import apply
from docta_orchestrator.realtext.stated import MIN_NAME, Index, mentions

from docta_eval.metrics import Gold, Key, Pred, ratio

# Reflection measures scored against required_sources, which real questions
# lack: they would read 0 or "late" for every model.
READING = ("reading_recall", "reading_precision", "early_conclusion_rate",
           "late_conclusion_rate", "dig_recovery_rate")  # fmt: skip


@dataclass(frozen=True)
class Probe:
    """One domain as the scorer needs it, read and chunked once."""

    world_id: str
    entities: dict[str, dict]
    questions: list[dict]
    texts: dict[str, str]
    index: Index


def is_domain(path: Path) -> bool:
    return Path(path).is_dir() and (Path(path) / "questions.jsonl").exists()


def load(where: Path) -> Probe:
    entities = named(read(where / "entities.jsonl"))
    after = apply(read(where / "articles.jsonl"), read(where / "redactions.jsonl"))
    texts = {s["source_id"]: s["text"] for s in sources(after)}
    return Probe(world(entities), {e["qid"]: e for e in entities},
                 read(where / "questions.jsonl"), texts, Index(after))  # fmt: skip


def read_real(where: Path) -> list[Gold]:
    return golds_of(load(where), where)


def golds_of(probe: Probe, where: Path) -> list[Gold]:
    golds = [gold(probe, q) for q in probe.questions]
    if not golds:
        raise ValueError(f"{where}: no question to score")
    return golds


def gold(probe: Probe, q: dict) -> Gold:
    names = frozenset(probe.entities[s]["name"] for s in q["subjects"])
    return Gold(world_id=probe.world_id, question_id=q["question_id"], primitive=q["primitive"],
                hops=q["hops"], answer=q["answer"] if q["answerable"] else None,
                required_sources=frozenset(), subjects=names, shape=q["shape"],
                absence=q["unanswerable"] or "answerable", also=stated(probe, q))  # fmt: skip


def stated(probe: Probe, q: dict) -> frozenset[str]:
    """Every key value the texts state for the year a question asks."""
    if not q["answerable"] or q["shape"] == "compare":
        return frozenset()
    e = probe.entities[q["subjects"][-1]]
    return frozenset(v for v in q["values"] if probe.index.stating(e["qid"], e["name"], [v]))


def supported(probe: Probe, golds: list[Gold], preds: dict[Key, Pred]) -> dict:
    questions = {q["question_id"]: q for q in probe.questions}
    right = backed = 0
    for g in golds:
        p = preds.get((g.world_id, g.question_id))
        if p is None or p.abstained or not g.accepts(p.answer):
            continue
        right += 1
        cited = [(sid, probe.texts.get(sid, "")) for sid in sorted(p.cited_sources)]
        rests = leans_on(questions[g.question_id], probe, p.answer)
        backed += all(backs(cited, probe.entities[qid], values) for qid, values in rests)
    return {"right": right, "supported": backed, "supported_rate": ratio(backed, right)}


def leans_on(q: dict, probe: Probe, answer: str) -> list[tuple[str, list[str]]]:
    """Each entity a right answer rests on, with the years that back it: a
    comparison, both sides' years; otherwise the year given, on the entity
    it belongs to (a chain's last)."""
    if q["shape"] == "compare":
        return [(s, [str(v) for v in asked(probe.entities[s])[1]]) for s in q["subjects"]]
    return [(q["subjects"][-1], [answer])]


def backs(cited: list[tuple[str, str]], entity: dict, values: list[str]) -> bool:
    # Same line as stated.py: a short name ("C") counts in its own article only.
    for sid, text in cited:
        own = sid.rsplit(":", 1)[0] == entity["qid"]
        named_here = len(entity["name"]) >= MIN_NAME and mentions(text, entity["name"])
        if (own or named_here) and any(mentions(text, v) for v in values):
            return True
    return False


def blank_reading(process: dict) -> dict:
    return {**process, **dict.fromkeys(READING)}
