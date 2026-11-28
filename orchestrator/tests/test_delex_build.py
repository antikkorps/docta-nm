"""Building a domain: every article rewritten with corpus-wide names, the
mapping kept apart for the leak check, the same bytes on every run."""

import json

from docta_orchestrator.delex import build

ART = {"qid": "Q1", "title": "Brask Bridge", "links": ["Varn River"], "url": "u", "sha256": "h",
       "sections": [{"path": ["Brask Bridge"], "paragraphs": ["Brask Bridge crosses the Varn "
                     "River, opened in 1901; the river under the bridge is 120 m wide."]}]}  # fmt: skip
ENT = {"qid": "Q1", "kind": "bridge", "label": "Brask Bridge", "title": "Brask Bridge",
       "aliases": [], "facts": {"opened": [1901], "length_m": [120],
       "crosses": [{"qid": "Q5", "label": "Varn River"}]}}  # fmt: skip


def write(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def test_a_domain_is_rewritten_with_its_mapping_apart(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    write(raw / "articles.jsonl", [ART])
    write(raw / "entities.jsonl", [ENT])
    (tmp_path / "names.txt").write_text("River\nKelda\nRosta\n")
    build.main([str(raw), "--out", str(out), "--names", str(tmp_path / "names.txt")])
    art = json.loads((out / "articles.jsonl").read_text())
    assert "Brask" not in json.dumps(art) and "Varn" not in json.dumps(art)
    assert "url" not in art and "sha256" not in art
    assert json.loads((out / "mapping.json").read_text())["names"] == {"Brask": "Kelda",
                                                                       "Varn": "Rosta"}  # fmt: skip
    first = (out / "articles.jsonl").read_bytes()
    build.main([str(raw), "--out", str(out), "--names", str(tmp_path / "names.txt")])
    assert (out / "articles.jsonl").read_bytes() == first
