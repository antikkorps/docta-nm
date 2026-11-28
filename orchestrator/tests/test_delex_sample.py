"""The hand-check sample: paragraphs drawn at random, the original beside
its rewriting, each domain in proportion to its size."""

from docta_orchestrator.delex import sample


def corpus(n, domain):
    para = "A paragraph long enough to be worth reading by hand, about a place. " * 4
    sections = [{"path": ["T"], "paragraphs": [para, "Short."]}]
    return [
        {"qid": f"Q{i}", "domain": domain, "title": "T", "sections": sections} for i in range(n)
    ]


def test_pairs_are_aligned_proportional_and_seeded():
    domains = {"mountains": (corpus(30, "mountains"), corpus(30, "mountains")),
               "bridges": (corpus(10, "bridges"), corpus(10, "bridges"))}  # fmt: skip
    picked = sample.draw(domains, n=8, seed=0)
    assert len(picked) == 8
    assert sum(p["domain"] == "mountains" for p in picked) == 6
    assert all(p["original"] == p["rewritten"] and len(p["original"]) > 200 for p in picked)
    assert picked == sample.draw(domains, n=8, seed=0)


def test_the_most_famous_subjects_open_the_sample_with_their_lead():
    # Lowest Wikidata ids are the best-known subjects: their lead paragraph
    # is where a phrasing ("the highest mountain on Earth") gives them away.
    domains = {"mountains": (corpus(30, "mountains"), corpus(30, "mountains")),
               "bridges": (corpus(10, "bridges"), corpus(10, "bridges"))}  # fmt: skip
    picked = sample.draw(domains, n=8, seed=0, famous=2)
    lead = [p for p in picked if p["famous"]]
    assert [p["id"] for p in lead] == ["bridges-Q0-0-0", "bridges-Q1-0-0",
                                       "mountains-Q0-0-0", "mountains-Q1-0-0"]  # fmt: skip
    assert len(picked) == 8 and len({p["id"] for p in picked}) == 8


def test_shares_add_up_to_the_size_asked():
    domains = {d: (corpus(k, d), corpus(k, d)) for d, k in
               (("universities", 100), ("mountains", 100), ("bridges", 62))}  # fmt: skip
    assert len(sample.draw(domains, n=50, seed=0, famous=3)) == 50
