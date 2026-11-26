"""A Wikipedia extract keeps its whole structure: sections, then paragraphs."""

from docta_orchestrator.realtext.structure import sections

EXTRACT = """Brask is a city in Ohio.
It was founded in 1820.


== History ==


=== Early years ===
The town grew around a mill.

== Geography ==
Brask lies at 210 metres.

== See also ==
Varnelle

== References =="""


def test_the_lead_comes_first_under_the_title():
    assert sections("Brask", EXTRACT)[0] == {
        "path": ["Brask"],
        "paragraphs": ["Brask is a city in Ohio.", "It was founded in 1820."],
    }


def test_each_section_keeps_its_path():
    paths = [s["path"] for s in sections("Brask", EXTRACT)]
    assert paths == [["Brask"], ["Brask", "History", "Early years"], ["Brask", "Geography"]]


def test_sections_without_prose_are_dropped():
    text = " ".join(p for s in sections("Brask", EXTRACT) for p in s["paragraphs"])
    assert "Varnelle" not in text


def test_a_skipped_level_still_nests_under_the_last_heading():
    text = "Lead.\n== A ==\n==== Deep ====\nBody."
    assert sections("T", text)[-1]["path"] == ["T", "A", "Deep"]
