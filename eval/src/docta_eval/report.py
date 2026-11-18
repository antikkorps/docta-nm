"""Console tables of the scores, one row per model (and per group)."""

COLUMNS = [
    ("accuracy", "acc"),
    ("justified_accuracy", "just acc"),
    ("citation_precision", "cite P"),
    ("citation_recall", "cite R"),
    ("abstention_precision", "abst P"),
    ("abstention_recall", "abst R"),
    ("abstention_f1", "abst F1"),
    ("false_abstention_rate", "bad abst"),
]
WIDTH = 8

REFLECTION = [
    ("reading_recall", "read R"),
    ("reading_precision", "read P"),
    ("mean_rounds", "rounds"),
    ("conclusion_rate", "concl"),
    ("early_conclusion_rate", "early"),
    ("late_conclusion_rate", "late"),
    ("searched_before_abstaining", "searched"),
    ("grounded_abstention_rate", "grounded"),
    ("dig_recovery_rate", "dug"),
]


def cell(value: float | None) -> str:
    # An undefined metric is shown as such, never as a misleading 0.
    return "—" if value is None else f"{value:.3f}"


def table(title: str, rows: list[tuple[str, dict]], columns=COLUMNS) -> str:
    names = [name for name, _ in rows]
    width = max(len(n) for n in [*names, title])
    header = f"{title:<{width}}  " + "  ".join(f"{h:>{WIDTH}}" for _, h in columns) + "  missing"
    lines = [header, "-" * len(header)]
    for name, s in rows:
        values = "  ".join(f"{cell(s[k]):>{WIDTH}}" for k, _ in columns)
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
    reflection = [(m, s["process"]) for m, s in results.items() if "process" in s]
    if reflection:
        parts.append(table("reflection", reflection, REFLECTION))
    return "\n\n".join(parts)
