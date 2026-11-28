"""The leak check: what of the real article survives its rewriting.

`python -m docta_orchestrator.delex.leaks data/delex/bridges` compares the
collected domain with `rewritten/` and writes `rewritten/leaks.json`; it
fails past the thresholds, so a leaky corpus never reaches training. Names
made only of common words ("Old Bridge") are listed, not counted against
the corpus: their real context is already gone."""

import json
import re
import sys
from pathlib import Path

from docta_orchestrator.delex.words import CALENDAR, ROMAN, SENTENCE_END, WORD, lowercase_words

# A real value left in one fact in fifty, or two unexplained capitalised
# words per thousand, means a rule is missing: fix it before training.
MAX_VALUE_RATE = 0.02
MAX_CAPITALISED_PER_1000 = 2.0


def pieces(article: dict) -> list[str]:
    return [t for s in article["sections"] for t in s["path"] + s["paragraphs"]]


def text(article: dict) -> str:
    return "\n".join(pieces(article))


def written(value) -> set[str]:
    return {str(value), f"{value:,}"}


def values(entity: dict) -> list:
    return [v for vs in entity["facts"].values() for v in vs if isinstance(v, (int, float))]


def capitalised(texts: list[str], invented: set[str], common: set[str]) -> list[str]:
    """Capitalised words in mid-sentence the rewriting left: real names it
    missed. Each text apart: a heading or a paragraph starts a sentence."""
    return [w for body in texts for w in mid_sentence(body, invented, common)]


def is_left(value, body: str) -> bool:
    """The value in any of its written forms, not inside a longer number."""
    return any(re.search(rf"(?<![\d.,]){re.escape(f)}(?!\d|[.,]\d)", body) for f in written(value))


def mid_sentence(body: str, invented: set[str], common: set[str]) -> list[str]:
    found = []
    for m in WORD.finditer(body):
        w = m.group(0)
        starts = not m.start() or SENTENCE_END.search(body[: m.start()])
        if starts or not w[0].isupper() or len(w) < 2:
            continue
        if (
            w not in invented
            and w.lower() not in common
            and w not in CALENDAR
            and not ROMAN.fullmatch(w)
        ):
            found.append(w)
    return found


def of_article(raw: dict, entity: dict, rewritten: dict, invented: set[str], common: set[str]):
    body = text(rewritten)
    left = [v for v in values(entity) if is_left(v, body)]
    names = [entity["label"], *entity.get("aliases", [])]
    subject = next(
        (n for n in names if len(n) > 2 and re.search(rf"\b{re.escape(n)}\b", body)), None
    )
    return {
        "values": left,
        "subject": subject,
        "capitalised": capitalised(pieces(rewritten), invented, common),
    }


def report(found: list[tuple[dict, int]], words: int) -> dict:
    """`found`: each article's leaks with its number of fact values."""
    leaked = sum(len(f["values"]) for f, _ in found)
    total = sum(n for _, n in found) or 1
    caps = sum(len(f["capitalised"]) for f, _ in found)
    out = {"articles": len(found), "value_rate": leaked / total,
           "capitalised_per_1000": caps / max(words, 1) * 1000,
           "subjects_left": sorted({f["subject"] for f, _ in found if f["subject"]})}  # fmt: skip
    out["passed"] = (out["value_rate"] <= MAX_VALUE_RATE
                     and out["capitalised_per_1000"] <= MAX_CAPITALISED_PER_1000)  # fmt: skip
    return out


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str]) -> int:
    domain = Path(argv[0])
    raw = {a["qid"]: a for a in read(domain / "articles.jsonl")}
    ents = {e["qid"]: e for e in read(domain / "entities.jsonl")}
    rewritten = read(domain / "rewritten/articles.jsonl")
    mapping = json.loads((domain / "rewritten/mapping.json").read_text(encoding="utf-8"))
    invented, common = set(mapping["names"].values()), lowercase_words(rewritten)
    found = []
    for a in rewritten:
        e = ents[a["qid"]]
        found.append((of_article(raw[a["qid"]], e, a, invented, common), len(values(e))))
    out = report(found, sum(len(text(a).split()) for a in rewritten))
    (domain / "rewritten/leaks.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "subjects_left"}))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
