"""Console tables of the scores, one row per model (and per group)."""

COLUMNS = [
    ("accuracy", "acc"),
    ("citation_precision", "cite P"),
    ("citation_recall", "cite R"),
    ("abstention_precision", "abst P"),
    ("abstention_recall", "abst R"),
    ("abstention_f1", "abst F1"),
    ("false_abstention_rate", "bad abst"),
]
WIDTH = 8


def cell(value: float | None) -> str:
    # An undefined metric is shown as such, never as a misleading 0.
    return "—" if value is None else f"{value:.3f}"


def table(title: str, rows: list[tuple[str, dict]]) -> str:
    names = [name for name, _ in rows]
    width = max(len(n) for n in [*names, title])
    header = f"{title:<{width}}  " + "  ".join(f"{h:>{WIDTH}}" for _, h in COLUMNS) + "  missing"
    lines = [header, "-" * len(header)]
    for name, s in rows:
        values = "  ".join(f"{cell(s[k]):>{WIDTH}}" for k, _ in COLUMNS)
        lines.append(f"{name:<{width}}  {values}  {s['missing']:>7}")
    return "\n".join(lines)


def render(results: dict[str, dict]) -> str:
    """Overall table, then the breakdowns by primitive and by hops."""
    parts = [table("model", [(m, s["overall"]) for m, s in results.items()])]
    for key, label in [("by_primitive", "primitive"), ("by_hops", "hops")]:
        rows = [
            (f"{m} · {group}", s)
            for m, scores in results.items()
            for group, s in scores[key].items()
        ]
        parts.append(table(f"model · {label}", rows))
    return "\n\n".join(parts)
