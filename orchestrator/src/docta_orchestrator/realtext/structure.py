"""Splits a Wikipedia plain-text extract into sections and paragraphs.

The whole structure is kept (section path, paragraph order), so any way of
serving excerpts (paragraph, sentence, neighbours) can be rebuilt later
without collecting again."""

import re

HEADING = re.compile(r"^(={2,6})\s*(.*?)\s*\1$")
# Lists of links and citations: no prose a question could rest on.
NO_PROSE = {
    "See also",
    "References",
    "External links",
    "Notes",
    "Further reading",
    "Sources",
    "Bibliography",
    "Citations",
    "Footnotes",
}


def sections(title: str, text: str) -> list[dict]:
    """[{"path": [title, "History", ...], "paragraphs": [...]}], lead first."""
    path, found, current = [title], [], []
    for line in (raw.strip() for raw in text.splitlines()):
        heading = HEADING.match(line)
        if heading:
            keep(found, path, current)
            depth = len(heading.group(1)) - 1  # "==" is the first level under the title
            path, current = path[:depth] + [heading.group(2)], []
        elif line:
            current.append(line)
    keep(found, path, current)
    return [s for s in found if not NO_PROSE & set(s["path"][1:])]


def keep(found: list[dict], path: list[str], paragraphs: list[str]) -> None:
    if paragraphs:
        found.append({"path": list(path), "paragraphs": paragraphs})
