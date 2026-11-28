"""The real-text probe's answer key never reaches a model: only the probe's
own package and the scorer read entities.jsonl, the questions' answers or
the cuts, and the runner sees each question's text through realtext.gold.

A text scan, so it sees what is written: the files by name, the package by
import (absolute or relative), the scorer's package (which reaches the key
in turn), and SQL reading the key's tables. A name computed at run time
(importlib, a joined path) would pass it unseen.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = [ROOT / p for p in ("orchestrator/src", "eval/src", "train/src")]
REALTEXT = "orchestrator/src/docta_orchestrator/realtext/"
# The delexicalisation corpus is collected with the probe's tools (fetcher,
# Wikidata, article records) and writes a key of its own; it never reads the
# probe's data, which test_delex_never_names_the_probe_data checks.
DELEX = "orchestrator/src/docta_orchestrator/delex/"
SCORER = "eval/src/docta_eval/real.py"
RUNNER = "orchestrator/src/docta_orchestrator/baseline.py"
# "entities" and "questions" are also keys of the worlds' records; the
# files' full names and "redactions" are the probe's alone.
KEY = re.compile(
    r"(entities|questions)\.jsonl|\bredactions\b"
    r"|^\s*(from|import)\s+\S*\brealtext\b|\bimport\s+realtext\b"
    r"|\bFROM\s+(facts|questions)\b"
)
SCORING = re.compile(r"^\s*(from|import)\s+docta_eval\b")


def lines_matching(pattern: re.Pattern, roots: list[Path]) -> dict[str, list[str]]:
    """Each module under `roots`, with its lines the pattern finds."""
    out = {}
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            name = path.relative_to(root.parents[1] if root in PACKAGES else root).as_posix()
            text = path.read_text(encoding="utf-8").splitlines()
            lines = [line.strip() for line in text if pattern.search(line)]
            if lines:
                out[name] = lines
    return out


def touching_the_key(roots: list[Path] = PACKAGES) -> dict[str, list[str]]:
    found = lines_matching(KEY, roots)
    return {n: lines for n, lines in found.items() if not n.startswith((REALTEXT, DELEX))}


def test_only_the_scorer_and_the_runner_reach_the_real_text_key():
    assert set(touching_the_key()) <= {SCORER, RUNNER}


def test_the_runner_sees_the_questions_through_gold_only():
    # gold.asked hands over text and ids: no answer, value or subject.
    lines = touching_the_key().get(RUNNER, [])
    assert lines == ["from docta_orchestrator.realtext.gold import asked"]


def test_nothing_outside_the_scorer_imports_it():
    # docta_eval.io reads a domain through docta_eval.real: a model or a
    # trainer importing it would reach the key without naming it.
    outside = lines_matching(SCORING, [ROOT / "orchestrator/src", ROOT / "train/src"])
    assert outside == {}


def test_the_guard_sees_every_way_it_names(tmp_path):
    sneaky = {
        "by_file.py": 'KEY = "data/real/cities/entities.jsonl"\n',
        "relative.py": "from ..realtext.load import facts\n",
        "built.py": 'names = ("articles", "redactions")\n',
        "sql.py": 'conn.execute("SELECT answer FROM questions")\n',
    }
    for name, text in sneaky.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert set(touching_the_key([tmp_path])) == set(sneaky)


def test_delex_never_names_the_probe_data():
    probe = re.compile(r"data/real\b|\bredactions\b|realtext\.(gold|probe|sample|load)\b")
    assert lines_matching(probe, [ROOT / DELEX]) == {}
