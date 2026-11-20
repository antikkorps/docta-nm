"""The rule's Python copy of the seen templates must track worldgen's Rust ones:
generate a small corpus and check the rule reads all of seen, none of ood.
A random corpus may miss a template, so each one also round-trips on its own."""

import shutil
import subprocess
from pathlib import Path

import pytest
from docta_orchestrator.jsonl import read_records
from docta_orchestrator.models.rule_evaluate import needs, read_sentence, sentences
from docta_orchestrator.models.templates import COMPARISONS, QUESTIONS, SENTENCES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, list[dict]]:
    # Fail, not skip: `just test` needs Rust anyway, and a skipped drift check
    # would let the copy rot unnoticed.
    assert shutil.which("cargo"), "cargo not installed: the drift check needs worldgen"
    out = tmp_path_factory.mktemp("corpus")
    subprocess.run(
        ["cargo", "run", "-q", "-p", "docta-worldgen", "--", "--seed", "7", "--worlds", "60"]
        + ["--out", str(out), "--split-ratio", "0.4,0.1,0.1,0.4"]
        # Templates only: trap sources state false values and attribute them.
        + ["--trap-ratio", "0"],
        cwd=ROOT,
        check=True,
    )
    return {
        split: [r for _, r in read_records(out / f"{split}.jsonl")]
        for split in ("train", "val", "test", "ood")
    }


def stated(world: dict, fact_ids: list[str]) -> list[tuple[str, str, str]]:
    names = {e["id"]: e["name"] for e in world["entities"]}
    facts = {f["id"]: f for f in world["facts"]}
    return sorted(
        (names[f["subj"]], f["pred"], names.get(f["obj"], str(f["obj"])))
        for f in (facts[i] for i in fact_ids)
    )


def test_every_seen_sentence_reads_back_to_exactly_its_facts(corpus):
    checked = 0
    for split in ("train", "val", "test"):
        worlds = {r["world_id"]: r for r in corpus[split] if r["type"] == "world"}
        for r in corpus[split]:
            if r["type"] == "source":
                read = sorted(t for s in sentences(r["text"]) for t in read_sentence(s))
                assert read == stated(worlds[r["world_id"]], r["facts"]), r["text"]
                checked += 1
    assert checked


def test_every_seen_question_is_understood_and_no_ood_one_is(corpus):
    def questions(split):
        return [r["text"] for r in corpus[split] if r["type"] == "question"]

    seen = questions("train") + questions("val") + questions("test")
    assert seen and all(needs(q) is not None for q in seen)
    ood = questions("ood")
    assert ood and all(needs(q) is None for q in ood)


def reached(world: dict, need: tuple[str, tuple[str, ...]]) -> list[str]:
    """The fact ids a need walks through in the world's ground truth."""
    ids = {e["name"]: e["id"] for e in world["entities"]}
    subject, path = ids[need[0]], []
    for pred in need[1]:
        fact = next(f for f in world["facts"] if (f["subj"], f["pred"]) == (subject, pred))
        path.append(fact["id"])
        subject = fact["obj"]  # an entity id when the chain goes on
    return path


def test_the_rule_needs_exactly_the_facts_an_answerable_question_requires(corpus):
    worlds = {r["world_id"]: r for r in corpus["test"] if r["type"] == "world"}
    answerable = [r for r in corpus["test"] if r["type"] == "question" and r["answerable"]]
    assert answerable
    for q in answerable:
        wanted = [f for n in needs(q["text"]) for f in reached(worlds[q["world_id"]], n)]
        assert sorted(wanted) == sorted(q["required_facts"]), q["text"]


def test_every_template_copy_round_trips():
    for pred, templates in SENTENCES.items():
        for t in templates:
            assert read_sentence(t.format(s="Kel Dorn", o="Varnelle")) == [
                ("Kel Dorn", pred, "Varnelle")
            ]
    for pred, templates in QUESTIONS.items():
        for t in templates:
            assert needs(t.format(s="Kel Dorn")) == [("Kel Dorn", (pred,))]
    for pred, t in COMPARISONS.items():
        assert needs(t.format(s="Kel Dorn", o="Brask")) == [
            ("Kel Dorn", (pred,)),
            ("Brask", (pred,)),
        ]
