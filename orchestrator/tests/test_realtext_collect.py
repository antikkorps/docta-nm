"""Collection is polite (named client, a pause, each answer asked once) and
every article records the exact revision it came from."""

import http.client
import io
import json

import pytest
from docta_orchestrator.realtext import collect
from docta_orchestrator.realtext.fetch import USER_AGENT, Fetcher, FetchError
from docta_orchestrator.realtext.wiki import article_record


class Opener:
    def __init__(self, payload):
        self.payload, self.requests = payload, []

    def __call__(self, request, timeout):
        self.requests.append(request)
        return io.BytesIO(json.dumps(self.payload).encode())


def test_the_fetcher_names_a_generic_client_and_asks_each_url_once(tmp_path):
    opener = Opener({"ok": 1})
    fetch = Fetcher(tmp_path, pause=0, opener=opener)
    assert fetch.json("https://x.org/api", {"b": "2", "a": "1"}) == {"ok": 1}
    assert fetch.json("https://x.org/api", {"a": "1", "b": "2"}) == {"ok": 1}
    assert len(opener.requests) == 1
    assert opener.requests[0].get_header("User-agent") == USER_AGENT
    assert "@" not in USER_AGENT  # no personal address goes out


def test_the_fetcher_pauses_between_requests(tmp_path):
    waits = []
    fetch = Fetcher(tmp_path, pause=1.0, opener=Opener({}), sleep=waits.append)
    fetch.json("https://x.org/a", {})
    fetch.json("https://x.org/b", {})
    assert len(waits) == 1 and 0 < waits[0] <= 1.0


PAGE = {
    "query": {
        "pages": [
            {
                "pageid": 7,
                "title": "Brask",
                "revisions": [{"revid": 123}],
                "extract": "Brask is a city.\n== History ==\nFounded in 1820.",
            }
        ]
    }
}


def test_an_article_keeps_its_revision_licence_and_structure():
    entity = {"qid": "Q20", "domain": "cities", "title": "Brask"}
    record = article_record(entity, PAGE)
    assert record["url"] == "https://en.wikipedia.org/w/index.php?oldid=123"
    assert record["licence"] == "CC BY-SA 4.0" and len(record["sha256"]) == 64
    assert record["sections"][1] == {
        "path": ["Brask", "History"],
        "paragraphs": ["Founded in 1820."],
    }


def test_a_missing_article_is_skipped():
    missing = {"query": {"pages": [{"title": "Nowhere", "missing": True}]}}
    assert article_record({"qid": "Q1", "domain": "cities", "title": "Nowhere"}, missing) is None


def test_collect_writes_the_answer_key_and_the_articles_apart(tmp_path, monkeypatch):
    entities = [{"qid": "Q20", "domain": "cities", "kind": "city", "title": "Brask"}]
    monkeypatch.setitem(collect.DOMAINS, "cities", lambda ask, limit: entities)
    monkeypatch.setattr(collect.Fetcher, "json", lambda self, url, params: PAGE)
    collect.main(["cities", "--out", str(tmp_path / "c"), "--cache", str(tmp_path / "k")])
    key = [json.loads(line) for line in (tmp_path / "c" / "entities.jsonl").open()]
    texts = [json.loads(line) for line in (tmp_path / "c" / "articles.jsonl").open()]
    assert key == entities and texts[0]["qid"] == "Q20"
    assert "facts" not in texts[0]  # what a model reads holds no answer key


class Raw(Opener):
    def __call__(self, request, timeout):
        self.requests.append(request)
        return io.BytesIO(self.payload)


def test_an_error_answer_is_refused_and_never_cached(tmp_path):
    fetch = Fetcher(tmp_path, pause=0, opener=Opener({"error": {"code": "maxlag"}}))
    with pytest.raises(FetchError, match="maxlag"):
        fetch.json("https://x.org/api", {})
    assert list(tmp_path.iterdir()) == []


def test_a_cut_answer_says_what_happened_and_is_never_cached(tmp_path):
    # The query service sends 200, then cuts the JSON when a query runs too long.
    fetch = Fetcher(tmp_path, pause=0, opener=Raw(b'{"results": {"bindings": [{"c'))
    with pytest.raises(FetchError, match="cut"):
        fetch.json("https://x.org/sparql", {})
    assert list(tmp_path.iterdir()) == []


def test_a_dropped_connection_is_a_fetch_error(tmp_path):
    def drops(request, timeout):
        raise http.client.IncompleteRead(b"")

    with pytest.raises(FetchError, match="connection"):
        Fetcher(tmp_path, pause=0, opener=drops).json("https://x.org/sparql", {})
