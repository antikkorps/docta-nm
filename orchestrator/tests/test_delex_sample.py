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
