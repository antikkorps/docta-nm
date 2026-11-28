"""The leak check: what of the real article survives its rewriting.

`python -m docta_orchestrator.delex.leaks data/delex/bridges` compares the
collected domain with `rewritten/` and writes `rewritten/leaks.json`; it
fails past the value threshold, so a leaky corpus never reaches training.
Names made only of common words ("Old Bridge") are listed, not counted:
their real context is already gone. Names the rules miss otherwise cannot
be counted by the same rules; the hand check (`delex.sample`) looks for them."""

import json
import re
import sys
from pathlib import Path

from docta_orchestrator.delex.numbers import NUMBER

# A real value left in one fact in fifty means a rule is missing: fix it
# before training.
MAX_VALUE_RATE = 0.02
# Values up to 31 read as days of a month, which the rewriting keeps.
SMALLEST = 32
# A run of three digits or more: what NUMBER should always have seen.
DIGITS = re.compile(r"\d{3,}")


def pieces(article: dict) -> list[str]:
    return [t for s in article["sections"] for t in s["path"] + s["paragraphs"]]


def text(article: dict) -> str:
    return "\n".join(pieces(article))


def values(entity: dict) -> list:
    return [v for vs in entity["facts"].values() for v in vs
            if isinstance(v, (int, float)) and (v >= SMALLEST or v != int(v))]  # fmt: skip


def occurrences(value, body: str) -> list[tuple[int, int]]:
    """Where `value` is written in `body`, plain or with thousands commas,
    found digit by digit: independent of the rewriter's own NUMBER rule."""
    found = []
    for form in {str(value), f"{value:,}"}:
        pattern = rf"(?<![\d.,]){re.escape(form)}(?!\d|[.,]\d)"
        found += [m.span() for m in re.finditer(pattern, body)]
    return found


def left_in(value, a: str, b: str) -> bool:
    """Whether paragraph `b`, the rewriting of `a`, still holds `value` where
    `a` had it. A spot NUMBER never matched was never rewritten: a leak if
    the value is still in `b`. A spot it matched pairs with the same-rank
    number of `b` (the rewriting keeps their order); another event's year
    shifted onto the real one elsewhere is a coincidence, not a leak."""
    seen, kept = list(NUMBER.finditer(a)), list(NUMBER.finditer(b))
    for start, end in occurrences(value, a):
        rank = next((i for i, m in enumerate(seen) if m.start() <= start and end <= m.end()), None)
        if rank is None:
            if occurrences(value, b):
                return True
        elif rank < len(kept) and kept[rank].group(0) == seen[rank].group(0):
            return True
    return False


def missed(article: dict) -> int:
    """Runs of three digits or more NUMBER did not see: its blind spots."""
    count = 0
    for body in pieces(article):
        spans = [m.span() for m in NUMBER.finditer(body)]
        count += sum(not any(s <= m.start() and m.end() <= e for s, e in spans)
                     for m in DIGITS.finditer(body))  # fmt: skip
    return count


def of_article(raw: dict, entity: dict, rewritten: dict) -> dict:
    pairs = list(zip(pieces(raw), pieces(rewritten), strict=False))
    left = [v for v in values(entity) if any(left_in(v, a, b) for a, b in pairs)]
    body = text(rewritten)
    names = [entity["label"], *entity.get("aliases", [])]
    subject = next(
        (n for n in names if len(n) > 2 and re.search(rf"\b{re.escape(n)}\b", body)), None
    )
    return {"values": left, "subject": subject, "missed": missed(raw)}


def report(found: list[tuple[dict, int]], words: int) -> dict:
    """`found`: each article's leaks with its number of fact values."""
    leaked = sum(len(f["values"]) for f, _ in found)
    total = sum(n for _, n in found) or 1
    out = {"articles": len(found), "value_rate": leaked / total,
           "missed_numbers_per_1000": sum(f["missed"] for f, _ in found) / max(words, 1) * 1000,
           "subjects_left": sorted({f["subject"] for f, _ in found if f["subject"]})}  # fmt: skip
    out["passed"] = out["value_rate"] <= MAX_VALUE_RATE
    return out


def read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str]) -> int:
    domain = Path(argv[0])
    raw = {a["qid"]: a for a in read(domain / "articles.jsonl")}
    ents = {e["qid"]: e for e in read(domain / "entities.jsonl")}
    rewritten = read(domain / "rewritten/articles.jsonl")
    found = []
    for a in rewritten:
        e = ents[a["qid"]]
        found.append((of_article(raw[a["qid"]], e, a), len(values(e))))
    out = report(found, sum(len(text(a).split()) for a in rewritten))
    (domain / "rewritten/leaks.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "subjects_left"}))
    return 0 if out["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
