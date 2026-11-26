# Design: the challenge step (JSONL v1)

Status: proposed, 2026-11-18. Decisions marked **(decided)** were made with
the maintainer; the rest is the plan to review.

## Why

The method is: collect, sort, **challenge**, dig, answer. Until now every
source told the truth, `reliability` was a random number with no meaning, and
nothing said where a source got its information. The Confront state had
nothing to arbitrate. Counting sources that agree is not enough either: many
copies of one false origin weigh as one source, not many. That is how false
consensus (fake news) fools a vote, and how repetition fools a model that
learned from frequency.

v1 makes the worlds contain disagreements, so the core can learn to weigh
sources, trace copies back to their origin, and say "the sources disagree and
I cannot settle it" when that is the honest answer.

## Three traps (decided)

Each trap targets one fact a question asks about (the *contested fact*).

| Trap | Sources about the contested fact | Expected behaviour |
|---|---|---|
| `contradiction` | one true source, one independent false source, clearly less reliable | answer the true value, cite the true source |
| `false_consensus` | one reliable primary source (true); one false *origin* plus 2 to 4 *copies* that attribute it in their text | answer the true value: copies count as one source |
| `undecidable` | two independent sources, equally reliable, opposite values | exit `unresolved_contradiction`, no answer |

## How the core can see a copy (decided: in the text)

A copy says where it got the claim, as an article citing a wire does:
"According to the Chronicle of Varnelle, Varnelle was founded in 1450."
Seen phrasings ("According to {t}, ", "As {t} reports, ") and one held-out
phrasing for `ood`, like every other template. `search()` keeps its four
fields: nothing about provenance leaks outside the text.

## Reliability (decided: indicative but noisy)

- Every source, true or false, has a reliability in 0.60–1.00: a value
  outside the truthful range would give a liar away without reading.
- False origins are drawn lower on average (0.60–0.75), so reliability helps
  but overlaps with true sources; sorting by reliability alone is not enough.
- Within a trap the rule must stay decidable: in `contradiction` the true
  source (0.80–1.00) is at least 0.20 more reliable; in `false_consensus` the
  primary source (0.85–1.00) is at least 0.10 above the false origin; in
  `undecidable` both are within 0.05 (inclusive), so no other trap can look
  undecidable. In a simple contradiction reliability does decide: that is the
  easy case the other two traps go beyond.
- Copies take a mid reliability (0.60–0.85): repeating a claim must not make
  it look better sourced than it is.

## JSONL v1

Every line carries `"schema":"v1"`. v1 is v0 plus:

- **source**: `false_claims` — ground truth of what it states wrongly,
  `[{"fact": "f3", "obj": 1450}]` (empty for most sources); `copy_of` — the
  source id it repeats, or `null`. Both are ground truth, never exposed.
- **question**: `trap` — `contradiction`, `false_consensus`, `undecidable` or
  `null`; `expected_exit` — `unresolved_contradiction` or `null`. For an
  `undecidable` question `answerable` is `false` and `missing` explains the
  disagreement. For the other traps `required_sources` lists the true source
  only: citing a copy or the false source costs citation precision.

The generator gains `--trap-ratio` (default 0.3: ceiling on the share of
answerable retrieve questions whose fact gets a trap; one trap per true source
and none on a fact another question uses, so about 23% get one, 5% of all
questions, and `just gen` reports the real count per split). Traps add
sources; they never remove the true one.
`--trap-ratio 0` gives worlds without traps.

## Guards (worldgen tests, extending `quality.rs` and `ood.rs`)

- Every trap is decidable as specified (reliability gaps, independence).
- No shortcut: false sources are not always the least reliable of a world;
  copies are not always shorter or longer; the contested fact's subject is not
  over-represented compared to other subjects.
- Attribution phrasings: the held-out one appears only in `ood`.
- Determinism and the split by world are unchanged.

## First cut: scope and known limits

- Traps target **retrieve** (1-hop) questions only; chains and comparisons
  over a contested fact come later.
- An `undecidable` question keeps its `retrieve` shape and becomes
  unanswerable; `trap` tells it apart from `spot_missing` questions, whose
  fact no source states.
- Trap sources hold a single sentence. Ordinary sources hold one to three, so
  a short source is slightly more often a trap source: a known shortcut, to
  remove by padding trap sources with true facts once that no longer breaks
  the "each fact stated once" guarantees.
- New sources are inserted at random places and every source is renumbered,
  so a source id never tells a trap source apart.
- Two shortcuts stay open, measured by tests that fail if they grow: a trapped
  subject is named by about 1.6 times more sources (false consensus: 7.1
  against 3.3), and a copy is always longer than its origin (it adds an
  attribution). Filler sources in v1.1 close both.
- An `undecidable` question takes an unanswerable retrieve slot of its world,
  which becomes a fresh answerable retrieve question, so each world keeps 9
  questions, 3 of them unanswerable, and every shape keeps its share.

## Database

`sources` gains `false_claims jsonb` and `copy_of text`; `questions` gains
`trap text` and `expected_exit text`. All ground truth: `search()` still
returns only `source_id`, `title`, `body`, `reliability`. `just reset-db`.

## Evaluation

Next to the existing scores, by trap:
- **accuracy and justified accuracy** on `contradiction` and `false_consensus`;
- **fooled rate**: share of trap questions answered with the false value;
- **right exit**: precision and recall of `unresolved_contradiction`;
- reflection: did it read both sides before deciding (from the traces).

## Baseline models

- `keyword` as today: takes whichever sentence fits best, blind to conflicts.
- `majority`: counts the sources behind each value and takes the most
  frequent — the naive vote that false consensus should fool.
- `challenge`: a hand-written Confront that groups copies with their origin,
  weighs independent sources by reliability, and exits
  `unresolved_contradiction` when the best two are within 0.05. It cites each
  winning group's origin (or best copy, if the origin was not read), and does
  not conclude on copies alone. Both rules read sentences with the seen
  templates: on `ood`, whose sentences are held out too, they fall back to
  keyword, like `rule_evaluate`. They are floors on seen phrasings; a trained
  Confront has to beat them on `ood`.

## Delivery, one pull request each

1. The v1 format end to end: worldgen (traps, attribution templates, guards),
   the JSONL reader, the database schema and the loader. One pull request,
   because CI runs the whole chain and the reader accepts one version only.
2. Evaluation: trap scores and exits.
3. Baseline models `majority` and `challenge`; the Confront state used.
