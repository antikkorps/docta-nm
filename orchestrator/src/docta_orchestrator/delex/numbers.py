"""Years shift, other numbers scale, the same way in the text and the facts.

One shift and one factor per article keep it coherent: order and gaps
between dates hold, ratios between quantities hold (metres and feet stay in
proportion), but no value is the real one."""

import re
from decimal import ROUND_HALF_UP, Decimal

MONTHS = ("January|February|March|April|May|June|July|August|September|October"
          "|November|December")  # fmt: skip
# A number with optional thousands commas and decimals, not inside a word;
# a comma after it is punctuation ("in 1786, the"), not a thousands mark.
# A decade's "s" ("the 1990s") or a unit glued to it ("8,849m") may follow.
UNITS = (
    r"(?:m|km|cm|mm|ft|mi|in|mph|kg|ha|sq|metres|meters|feet|miles|kilometres"
    r"|kilometers)(?:²|³)?"
)
# Ranks ("11th") scale like quantities; centuries stay, as the years they
# cover move by less than a century.
ORDINAL = r"(?:st|nd|rd|th)"
ORDINAL_AFTER = re.compile(rf"^{ORDINAL}\b")
# An ordinal in a list or range of centuries stays: "14th and 15th
# centuries", "4th–5th Century".
CENTURY = re.compile(
    r"^(?:st|nd|rd|th)(?:[\s,–—-]+(?:and|or|to)?[\s,–—-]*\d+(?:st|nd|rd|th))*"
    r"[\s–—-]+(?:century|centuries|millennium|millennia)\b",
    re.IGNORECASE,
)
# A scaled rank keeps its number's suffix: "1st" scaled to 2 reads "2nd".
RANK = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b")
NUMBER = re.compile(
    # Not inside a word or a longer number; after "c." (circa) it is one.
    rf"(?<![\w,])(?<!\d\.)(\d{{1,3}}(?:,\d{{3}})+|\d+)(\.\d+)?(?!(?!s\b|{UNITS}\b|{ORDINAL}\b)\w|,\d|\.\d)"
)
# A unit after a number makes it a quantity, even one that reads like a year.
UNIT_AFTER = re.compile(rf"^\s?{UNITS}\b")
# Left alone: days next to a month ("August 8", "8 August") and percentages
# would read wrong once scaled; ordinals ("1st") never match NUMBER.
DAY = re.compile(rf"(?:{MONTHS})\s+$")
# The first day of a range ("8–10 August") is a day too.
DAY_FIRST = re.compile(
    rf"^(?:st|nd|rd|th)?(?:\s*[–-]\s*\d{{1,2}}(?:st|nd|rd|th)?)?\s+(?:of\s+)?(?:{MONTHS})\b"
)
PERCENT = re.compile(r"^\s*(%|percent\b)")


def suffix(n: int) -> str:
    if n % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def is_year(whole: str, decimals: str | None) -> bool:
    return decimals is None and "," not in whole and 1000 <= int(whole) <= 2030


def year(value: int, shift: int) -> int:
    return value + shift


def scaled(value: Decimal, factor: float) -> Decimal:
    """Scaled and rounded half up to as many decimals as `value` has. Exact
    decimals for the key and the text alike: a float would round 1402.5
    one way in a fact and the other in the sentence that states it."""
    places = max(-value.as_tuple().exponent, 0)
    step = Decimal(1).scaleb(-places)
    return (value * Decimal(str(factor))).quantize(step, rounding=ROUND_HALF_UP)


def quantity(value: float, factor: float) -> float:
    """A fact's value scaled as its stating text is."""
    out = scaled(Decimal(str(value)), factor)
    return int(out) if out == out.to_integral_value() and isinstance(value, int) else float(out)


def written(value: Decimal, like: str) -> str:
    """`value` in the form of `like`: thousands commas kept or not."""
    return f"{value:,}" if "," in like else str(value)


def rewrite(text: str, shift: int, factor: float) -> str:
    def one(m: re.Match) -> str:
        whole, decimals = m.group(1), m.group(2)
        before, after = text[: m.start()], text[m.end() :]
        is_day = len(whole) <= 2 and (DAY.search(before) or DAY_FIRST.match(after))
        if is_day or PERCENT.match(after):
            return m.group(0)
        if ORDINAL_AFTER.match(after):
            # "1st", "2nd", "3rd" mostly order events ("the 1st ascent"): kept.
            if CENTURY.match(after) or decimals or "," in whole or int(whole) < 4:
                return m.group(0)
            return str(max(1, int(scaled(Decimal(whole), factor))))
        if is_year(whole, decimals) and not UNIT_AFTER.match(after):
            return str(year(int(whole), shift))
        if after.startswith("s"):  # "20s": an age or a decade without century
            return m.group(0)
        value = Decimal(whole.replace(",", "") + (decimals or ""))
        return written(scaled(value, factor), m.group(0))

    return RANK.sub(lambda m: m.group(1) + suffix(int(m.group(1))), NUMBER.sub(one, text))
