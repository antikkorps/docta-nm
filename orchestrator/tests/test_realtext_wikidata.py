"""Wikidata answers become the probe's answer key, never what a model reads."""

from docta_orchestrator.realtext.wikidata import bindings, by_qid_order, qid, title_of, year


def test_bindings_keep_values_only():
    payload = {"results": {"bindings": [{"c": {"type": "uri", "value": "x"}}]}}
    assert bindings(payload) == [{"c": "x"}]


def test_qid_and_title():
    assert qid("http://www.wikidata.org/entity/Q3415") == "Q3415"
    assert (
        title_of("https://en.wikipedia.org/wiki/Saint_Paul%2C_Minnesota") == "Saint Paul, Minnesota"
    )


def test_year_reads_dates_before_the_common_era():
    assert year("-0386-01-01T00:00:00Z") == -386
    assert year("1820-05-01T00:00:00Z") == 1820


def test_order_is_by_number_not_by_text():
    assert by_qid_order(["Q100", "Q9", "Q9", "Q20"]) == ["Q9", "Q20", "Q100"]
