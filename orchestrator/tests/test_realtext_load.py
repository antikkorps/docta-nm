"""Loading the probe: the texts after the cuts become sources a model can
search, the key and the questions stay ground truth, in a `real` split."""

from docta_orchestrator.realtext.load import batch, chunks
from docta_orchestrator.search import search
from test_realtext_questions import ARTICLES, CITIES

QUESTIONS = [{"question_id": "q00000", "body": "In what year was Varna founded?", "hops": 1,
              "shape": "retrieve", "primitive": "retrieve", "answerable": True, "answer": "1749",
              "values": ["1749"], "missing": None, "subjects": ["Q1"]}]  # fmt: skip
CUTS = [{"article": "P1", "sentence": "In 1802 the mill burned.", "subject": "Q2"}]


def corpus():
    return batch("cities", CITIES, ARTICLES, CUTS, QUESTIONS)


def test_a_long_paragraph_is_cut_at_sentence_ends():
    long = " ".join(f"Sentence {i} has exactly six words." for i in range(30))
    parts = chunks(long, limit=20)
    assert all(len(p.split()) <= 20 for p in parts) and " ".join(parts) == long
    one = "A single sentence far longer than the limit stays whole."
    assert chunks(one, limit=5) == [one]  # cutting inside it would break what it says


def test_a_blank_paragraph_gives_no_source():
    assert chunks("   ") == []


def test_sources_are_the_texts_after_the_cuts_with_no_ground_truth():
    b = corpus()
    assert (b.world["world_id"], b.world["split"]) == ("real_cities", "real")
    bodies = [s["text"] for s in b.sources]
    assert not any("mill burned" in t for t in bodies)
    assert all(s["facts"] == [] and s["copy_of"] is None for s in b.sources)
    assert {s["reliability"] for s in b.sources} == {0.7}
    assert b.sources[0]["title"] == "Q1"  # the section path, article first


def test_the_key_goes_to_facts_and_questions_keep_their_answers():
    b = corpus()
    assert ("Q1", "inception", 1749) in {(f["subj"], f["pred"], f["obj"]) for f in b.world["facts"]}
    q = b.questions[0]
    assert (q["text"], q["answer"], q["required_sources"]) == (
        "In what year was Varna founded?", "1749", [])  # fmt: skip


def test_search_finds_a_paragraph_in_the_real_split(conn):
    from docta_orchestrator.load import load_world

    load_world(conn, corpus())
    hits = search(conn, "real_cities", "Varna established")
    assert hits and "1749" in hits[0].body
    split = conn.execute("SELECT split FROM worlds WHERE world_id = 'real_cities'").fetchone()
    assert split == ("real",)


def test_the_world_is_named_after_the_domain_not_the_folder(tmp_path, monkeypatch):
    import json

    from docta_orchestrator.realtext import load

    folder = tmp_path / "copy-of-cities"
    folder.mkdir()
    rows = {"entities": [{**e, "domain": "cities"} for e in CITIES], "articles": ARTICLES,
            "redactions": CUTS, "questions": QUESTIONS}  # fmt: skip
    for name, records in rows.items():
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    loaded = []
    monkeypatch.setattr(load, "store", loaded.append)
    assert load.main([str(folder)]) == 0
    assert loaded[0].world["world_id"] == "real_cities"
