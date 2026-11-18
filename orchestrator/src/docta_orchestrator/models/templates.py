"""The seen phrasings of worldgen/src/templates.rs, copied by hand for the rule.

Only the seen ones: the rule must know exactly what a core trained on
train/val sees, never the held-out ood phrasings. tests/test_rule_templates.py
checks this copy against a corpus worldgen generates.
"""

SENTENCES = {
    "founded_in": [
        "{s} was founded in {o}.",
        "The founding of {s} dates back to {o}.",
        "{s} has existed since {o}.",
    ],
    "population": [
        "{s} has a population of {o}.",
        "The last census counted {o} inhabitants in {s}.",
    ],
    "based_in": [
        "{s} is based in {o}.",
        "{s} keeps its seat in {o}.",
        "The halls of {s} stand in {o}.",
    ],
    "born_in": ["{s} was born in {o}.", "{s} came into the world in {o}."],
    "lives_in": ["{s} lives in {o}.", "{s} resides in {o}.", "{o} is home to {s}."],
    "member_of": ["{s} is a member of {o}.", "{s} belongs to {o}."],
    "elevation": [
        "{s} lies at an elevation of {o} metres.",
        "{s} sits {o} metres above sea level.",
    ],
    "members": ["{s} has {o} members.", "The rolls of {s} list {o} members."],
    "height": ["{s} is {o} centimetres tall.", "{s} stands {o} cm tall."],
}

QUESTIONS = {
    "founded_in": "In what year was {s} founded?",
    "population": "What is the population of {s}?",
    "based_in": "In which city is {s} based?",
    "born_in": "In what year was {s} born?",
    "lives_in": "In which city does {s} live?",
    "member_of": "Which guild does {s} belong to?",
    "elevation": "At what elevation, in metres, does {s} lie?",
    "members": "How many members does {s} have?",
    "height": "How tall is {s}, in centimetres?",
}

COMPARISONS = {
    "founded_in": "Which was founded earlier, {s} or {o}?",
    "population": "Which has the larger population, {s} or {o}?",
    "born_in": "Who was born earlier, {s} or {o}?",
    "elevation": "Which lies higher, {s} or {o}?",
    "members": "Which has more members, {s} or {o}?",
    "height": "Who is taller, {s} or {o}?",
}

# The first hop of a chain: only predicates whose object is an entity can lead
# to a second fact ("the city where X lives" → that city's founding year).
CHAIN_LINKS = {
    "based_in": "the city where {s} is based",
    "lives_in": "the city where {s} lives",
    "member_of": "the guild that {s} belongs to",
}
