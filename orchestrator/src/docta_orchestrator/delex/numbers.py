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
CENTURY = re.compile(r"^(?:st|nd|rd|th)[\s-]+(?:century|centuries|millennium)\b")
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
DAY_FIRST = re.compile(rf"^(?:\s*[–-]\s*\d{{1,2}})?\s+(?:{MONTHS})\b")
PERCENT = re.compile(r"^\s*(%|percent\b)")


def suffix(n: int) -> str:
    if n % 100 in (11, 12, 13):
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def is_year(whole: str, decimals: str | None) -> bool:
    return decimals is None and "," not in whole and 1000 <= int(whole) <= 2030


def year(value: int, shift: int) -> int:
    return value + shift


def quantity(value: float, factor: float) -> float:
    """Scaled and rounded to as many decimals as the value had."""
    places = -Decimal(str(value)).as_tuple().exponent
    scaled = (Decimal(str(value)) * Decimal(str(factor))).quantize(
        Decimal(1).scaleb(-max(places, 0)), rounding=ROUND_HALF_UP
    )
    return int(scaled) if places <= 0 else float(scaled)


def written(value: float, like: str) -> str:
    """`value` in the form of `like`: thousands commas, decimals kept."""
    decimals = len(like.split(".")[1]) if "." in like else 0
    text = f"{value:,.{decimals}f}" if "," in like else f"{value:.{decimals}f}"
    return text


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
            return str(max(1, int(quantity(int(whole), factor))))
        if is_year(whole, decimals) and not UNIT_AFTER.match(after):
            return str(year(int(whole), shift))
        if after.startswith("s"):  # "20s": an age or a decade without century
            return m.group(0)
        value = float(whole.replace(",", "") + (decimals or ""))
        return written(quantity(value, factor), m.group(0))

    return RANK.sub(lambda m: m.group(1) + suffix(int(m.group(1))), NUMBER.sub(one, text))
