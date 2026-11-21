"""The seen phrasings of worldgen's templates and grammar, for the rules.

Only the seen ones: the rule must know exactly what a core trained on
train/val sees, never the held-out ood phrasings. worldgen writes them to
templates_seen.json (`just templates`); a worldgen test keeps that file in
step, and tests/test_rule_templates.py checks it against a generated corpus.
"""

import json
import re
from pathlib import Path

_SEEN = json.loads((Path(__file__).with_name("templates_seen.json")).read_text(encoding="utf-8"))

SENTENCES: dict[str, list[str]] = _SEEN["sentences"]
QUESTIONS: dict[str, list[str]] = _SEEN["questions"]
COMPARISONS: dict[str, str] = _SEEN["comparisons"]
# The first hop of a chain: only predicates whose object is an entity can lead
# to a second fact ("the city where X lives" → that city's founding year).
CHAIN_LINKS: dict[str, str] = _SEEN["chain_links"]


def _capitalised(template: str) -> set[str]:
    return set(re.findall(r"\b[A-Z][a-z]+\b", re.sub(r"\{[so]\}", "", template)))


# A capitalised word a seen template, attribution or title writes ("Records",
# "It", "According", "Notes") is a template word, never a name: worlds' names
# are invented and spell none.
TEMPLATE_WORDS: frozenset[str] = frozenset(
    w
    for kind in (SENTENCES, QUESTIONS)
    for templates in kind.values()
    for t in templates
    for w in _capitalised(t)
) | frozenset(
    w
    for t in [*COMPARISONS.values(), *_SEEN["attributions"], *_SEEN["titles"]]
    for w in _capitalised(t.replace("{t}", "").replace("{x}", ""))
)
