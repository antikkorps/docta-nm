# Changelog

All notable changes to Docta-nm are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html): while the version
is `0.0.x`, anything may change between releases.

Every pull request that a user or contributor would notice adds its line under
**Unreleased**. A release moves those
lines under a new version, and its GitHub Release carries the reference scores.

## [Unreleased]

### Added

- **Delex leak check** (`python -m docta_orchestrator.delex.leaks
  data/delex/bridges`), and the rules it led to: a word is a name when the
  corpus writes it capitalised mid-sentence more often than lowercase
  ("China", not "Bridge"), acronyms included, sentence starts read past
  abbreviations ("Mt. Hood"), months and small Roman numerals excepted;
  "c.1901" and glued units ("120mph") count as numbers. The check finds
  each fact value digit by digit, independently of the rewriter: on the
  three domains none stays in place (fails past 2%). Subject names made of
  common words ("Old Bridge") are listed for the hand check (`delex.sample`:
  the leads of each domain's best-known subjects, then random paragraphs).
- **Delexicaliser** (`python -m docta_orchestrator.delex.build
  data/delex/bridges --out data/delex/bridges/rewritten --names
  data/delex/names.txt`, names from `docta-worldgen --emit-names`): every
  word of a linked or named entity that the corpus never writes in lowercase
  becomes an invented name, the same across the domain; years shift back by
  30 to 150 years per article, other numbers scale by one factor per
  article; the Wikidata facts change alike. The mapping is written apart,
  for the leak check.
- **Delexicalisation corpus, collected** (`python -m
  docta_orchestrator.delex.collect bridges --out data/delex/bridges`): the
  1,000 universities, mountains and bridges of lowest Wikidata id, each with
  every stated value, its aliases and related entities, and its article
  pinned to its revision with every internal link (the spans step 2 will
  replace). Kept local, never trained on as is.
- **Design of delexicalised training text** (`docs/design-delex.md`): real
  Wikipedia sentences from domains outside the probe, with names and values
  replaced by invented ones, so the core learns real English without real
  facts in its weights.
- **Roadmap and handoff** (`docs/ROADMAP.md`): where each chantier stands,
  what comes next, the open issues, how we work, and what to rebuild or copy
  when moving to another machine.
- **Real text, frozen** (`docta_orchestrator.realtext.collect`): 200 cities
  of the United States and 150 programming languages from Wikipedia, split
  into sections and paragraphs and pinned to their revision, with a Wikidata
  answer key kept apart. Collected once, never trained on: it measures how
  far the method carries to real English.
- **Questions on real text** (`python -m docta_orchestrator.realtext.probe
  data/real/cities`): from
  the answer key, years only (a city founded, a language released, a person
  born), as retrieve, chain and compare, in the worlds' own wordings where a
  relation exists there and tagged `new` where it does not. A question is
  unanswerable when no text states the value for its subject (its own
  article, or a sentence naming it), never because the key lacks it. To
  reach a third on every shape, the sentences stating some facts are cut
  (`redactions.jsonl`, applied when loading; `articles.jsonl` keeps the text
  as collected), and each unanswerable question says whether its absence is
  natural or made.
- **Real text in the database** (`python -m docta_orchestrator.realtext.load
  data/real/cities`): one world per domain (`real_cities`), its sources the
  paragraphs after the cuts (cut between sentences past 120 words, one
  reliability for all), the key in `facts`. `worlds.split` accepts `real`:
  run `just reset-db` before loading. `just load --prune` leaves the probe in
  place.
- **Real-text labels fixed from the hand check** (150 questions checked by
  Franck: answerable 8% wrong, 20% with the unsure ones; natural absence 8%;
  redacted 0%): sentences no longer break after an initial ("John G.
  Kemeny"), a chain's link may name a person by surname (never "Jr."), and
  a comparison naturally lacking an answer is dropped when its order reads
  off the two articles' other dates (20 of 48 such in cities, 10 of 62 in
  languages; redacted ones, all found right, are kept). A year stated for
  another event (named, incorporated) still counts: a known limit.
- **Hand-check sample** (`python -m docta_orchestrator.realtext.sample
  data/real/cities data/real/languages --out data/real/sample.jsonl`): so
  many questions of each kind (answerable, natural, redacted), drawn with a
  seed, each with the sentences that state its year, a chain's link, the
  cuts, and for an absence the years its article holds, so the label error
  can be reported next to the scores.
- **`just real cities`** (or `languages`): cuts and asks, loads, runs the
  baselines and the trained cores on the domain, and scores them; `just
  steps` shows a column for each domain scored.
- **Real text through the loop, scored**: `python -m docta_orchestrator.baseline`
  and `python -m docta_eval` take a prepared domain folder (`data/real/cities`)
  as well as a split. Scores add breakdowns by shape and by kind of absence;
  the worlds' fact citations show `—`, and *supported* (a cited paragraph
  holds the year and is the entity's own article or names it) stands in for
  them, weaker and labelled so; so do the reflection measures that need
  them. Any value the texts state for the year asked is a right answer.
  A test keeps the answer key out of every
  module but the probe's own, the scorer, and the runner's question texts.
- **The method, step by step** (`just steps`): one row per step (search,
  judge, read, dig, confront, conclude, human) and per measure, on test, `val`
  and `ood` side by side, with the change since the last release's snapshot
  in `scores/`, kept in the repository. `just human` now also saves its scores
  (`predictions/<split>/human/scores.json`).
- **Overlap flag for the Evaluate core** (`just fit --overlap`): each excerpt
  word the question also holds is marked, so the core no longer has to pair
  question and sentence templates to see the subject is there. Full data, 6
  epochs, 3 seeds: balanced `val` accuracy 80.6–84.1% (median 83.3%) against
  68.8–76.7% without; on `ood`, measured once, median 82.2% against 77.1% for
  the 0.0.2 core, and the checkpoint chosen on `val` beforehand wrongly abstains
  on 4.9% of answerable examples against 23.1%. Seeds still vary on `ood`
  (75.5–84.0%), and a guild chain wording stays near chance on all of them.

### Changed

- **Delex after the hand check** (50 paragraphs checked by Franck: 45
  clean, 5 with something real): words of scripts without letter case
  (Korean, Chinese) and acronyms in capitals ("SKY") are now names, and
  ranks from 4th up scale like other numbers (centuries stay). Four of the
  five leaks are gone; a name made of common words ("Castle Rock") stays,
  a known limit.
- **Search weighs query words by rarity** (`search()`): a source scores the
  sum of BM25's idf, within its world, of the question words it holds;
  `ts_rank` then `source_id` break ties. Unweighted, "in", "was" and "year"
  made the chattiest passage win. On the real-text probe the answer reaches
  the five excerpts for 53% of answerable city retrieve questions (32%
  before; languages 45%, 47% before); on the invented worlds' test and `ood`
  every baseline's accuracy moves by under a point.

- `just fit-read` warms up over 4000 steps by default, like `just fit`: over
  five seeds its `val_exact` spread over 8 points instead of 24, at a median
  3.9 points lower (59.1% against 63.0%), so runs can be compared.
- **Pronouns, palier 2 begins**: a source names a subject again by pronoun
  right after a sentence about it ("Kel Dorn was born in 1950. She lives in
  Brask."): he, she or singular they (45/45/10, drawn from the world seed and
  the person, never from the name, the verb agreeing with "they"), it for a
  city or a guild, as subject, object (him, them) or owner (her home, its
  ranks). A core must follow the text to know whom a sentence is about.
  Trap sources always name their subject. The rules read a pronoun as the
  subject of the sentence before, and find a sentence's templates by its
  closing words (the pronoun forms bring the seen templates to tens of
  thousands). Every split changes: rerun `just gen 42 20000`, `just load`,
  then retrain.
- **A richer language, palier 1**: each relation in 9 or 10 seen clauses
  (was 4 or 5), 30 frames (was 8), eight asides (was one), 16 titles (was
  8), three or more wordings per question, and descriptions that state and
  imply no fact ("Varnelle, a town of some renown, ..."): about 200 distinct
  template words (was about 90). No two relations share their distinctive
  words, and a frame never repeats a word of its aside. The clauses held out for val and ood are unchanged, still
  built from seen words. Every split changes: rerun `just gen 42 20000`,
  `just load`, then retrain.
- **JSONL v2**: each source carries the template that wrote each of its
  sentences (`phrasings`), each question its own (`template`): ground truth
  for diagnosis by wording, never a model input. Readers accept v2 only.
- `just fit` defaults to `--warmup 4000 --lr 1e-3 --dropout 0.1` (was no
  warmup, `--lr 3e-4`, no dropout): over three seeds on `val`, these got every
  Evaluate run off the shape-alone plateau.
- `just fit-read` keeps the epoch with the best `val_exact` instead of the
  last: on `val`'s held phrasings the Read core peaks before the end (65.6% at
  epoch 3, 61.5% at epoch 4 on one seed).
- `just fit` keeps the epoch with the best `val_accuracy` instead of the
  lowest `val_loss` (ties go to the lower loss), the score that is reported:
  on palier 1 an overconfident core's loss rose while it got more right
  (seed 2: 55.2% at the lowest loss, epoch 1; 71.3% at epoch 6).

### Fixed

- `search` refuses a `k` below 1 instead of reading `LIMIT 0` as "nothing
  found" or failing in SQL; Confront ties a copy to the read source that
  bears its credited title and says what it repeats, so two sources sharing
  that title can no longer turn the copy into a second voice (#18).
- `--max-minutes` no longer starts an epoch that would likely end past the
  budget (judged on the last epoch's length; the first always runs), where
  one long epoch could overrun it by far; an epoch is still never cut half-way, so `--resume`
  goes on exactly. `just fit` reports missing data, and both trainers an
  unwritable `--out`, with a message instead of a traceback (#19).
- `just fit` and `just fit-read` refuse an empty train or val split at once,
  saying how to get data, instead of failing on a division by zero after
  loading (#12).
- `just gen` stops with the world and the slot when a question slot finds
  no question, instead of writing a world with fewer than nine questions
  (#15). No world of `just gen 42 20000` is affected: its output is
  unchanged, byte for byte.
- `--trap-ratio` is documented as what it is, a ceiling: about 23% of
  answerable retrieve questions get a trap at the default 0.3 (5% of all
  questions). `just gen` now reports, per split, the worlds, the questions
  and how many got a trap (#16). The data is unchanged.
- Training no longer cuts an example longer than the core reads: a cut
  excerpt could lose the sentence its label or answer rests on. `just fit`
  and `just fit-read` now refuse such data, and the cores read up to 576
  tokens (was 320, which cut 1.7% of Evaluate examples on 0.0.2's data and
  13% on the richer language's). A checkpoint keeps the length it was
  trained with, and `--resume` keeps it too (#13).

## [0.0.2] - 2026-11-25

Trained cores take over steps of the method: the Evaluate core judges whether
what was read is enough, the Read core points at the answer and at the next
name to search, and `core_read` beats every baseline on test. `ood` is made
honest (held-out grammar, chain link wording and combinations) and `val` holds
phrasings of its own, so settings are chosen without opening `ood`; on `ood`
the cores remain far from test. A human-correction protocol measures how the
loop takes a hint mid-way. From this version Docta-nm is dual-licensed (AGPL
v3.0 or a commercial licence).

Training note, measured on `val` over three seeds: with `--warmup 1000 --lr
1e-3`, two of four Evaluate seeds stalled on the shape-alone plateau (val
about 0.60). `--warmup 4000 --lr 1e-3 --dropout 0.1` got all three seeds
tried off it (val accuracy 68.8%, 73.5%, 76.7% after 6 epochs); `--lr 3e-4`
did too, but with a rising val loss on two seeds (67.1–76.8%).

### Added

- **`val` held out from train**: `val` is written in phrasings of its own
  (one sentence clause, question, comparison and chain link per predicate),
  never in train, test or `ood`, so settings are chosen for generalisation
  without opening `ood`. Train, test and `ood` are byte-identical; `val`
  changes, so rerun `just gen && just load`.
- `core_bridge` (experimental, in `just core` once the Read core exists):
  no rule splits a chain; the Read core points at the bridge, code searches
  it, Evaluate judges the whole question. Evaluate and Read data now teach
  the step after the dig. Evaluate `ood` AUC 0.818 (was 0.635–0.753), but
  `core_bridge` chains 27.9% on test (`core_read` 69.3%), 6.9% on `ood`
  (0.2% justified for `core_read`); a subject check keeps a wrong hint from
  fooling it (41% → 0). A measured negative result.
- **Read core with two pointers** (experimental): one for the answer, one
  for the name to search next, on one encoding. Single seed: `core_bridge`
  chains 47.7% on test (27.9% with one pointer), 9.9% on `ood` (6.9%),
  `fooled` still 0. Read data and checkpoints from before are refused:
  rerun `just read-data && just fit-read`.
- **Human-correction protocol** (`just human`): each answerable chain runs
  with no hint, the right bridge and a wrong one said mid-way, through a
  human channel only Plan hears. Scores: accuracy per variant, `heard`,
  `reoriented`, `recovered` and `fooled`.
- **Read core** and `core_read` in `just core`: `just read-data` and
  `just fit-read` train a core that points at the answer in the excerpts;
  code copies it and cites its source, and compares values itself. On test,
  `core_read` answers 76.9% (`core_chain` 65.3%, `dig` 68.1%) with citation
  precision 0.91–0.94; `ood` stays at 26–32%, held back by Evaluate's abstentions.
- `core_chain` in `just core`: the orchestrator splits a two-hop question
  into one-hop questions and the trained core judges each. Chains go from
  none answered to 57% on test, abstention F1 from 0.71 to 0.77.
- `just fit --resume --epochs N` adds N epochs to the last run
  (`models/core_evaluate/last.pt`) instead of starting from scratch.
- **Trap scores**: a `model · trap` table and a `traps` entry in the JSON
  scores: accuracy and justified accuracy on decidable traps, how often a
  model was fooled, and the precision and recall of the
  `unresolved_contradiction` exit.
- **Confront baselines**: `majority` (one vote per source, fooled by false
  consensus) and `challenge` (copies count with their origin, independent
  sources weighed by reliability, `unresolved_contradiction` when too close).

- **External contributions**: `.github/CODEOWNERS` routes changes to
  tooling paths (CLAUDE.md, CI, justfile, dependencies, database schema) to
  the maintainer; CONTRIBUTING.md states the rules for AI-assisted
  contributions and that repository text is never an instruction to an AI.

### Changed

- **Dual licence**: Docta-nm moves from Apache-2.0 to the GNU AGPL v3.0
  (`AGPL-3.0-only`), with a commercial licence for use the AGPL does not
  allow. Contributions now come under a Contributor License Agreement
  (`CLA.md`), accepted once by ticking its box in a pull request. What was
  published before stays available under Apache-2.0.
- Digging searches for the bridge by its name alone, not with the question's
  other words, which brought back sources about anything near it. On test
  (grammar data, one Evaluate seed), `core_read` goes from 77.9% to 83.0%
  (chains 71.3% to 86.8%), `dig` from 67.4% to 68.3%; the right human hint
  now recovers 60% of `core_read`'s failed chains (was under 1%), `fooled` 0.
- **Sentences come from a combinatorial grammar**: each relation clause in
  eight frames, with or without a parenthetical after its subject, 30–48
  structures per predicate instead of 4–5. `ood` holds out (clause, form)
  pairs on top of its held-out clause. All data changes: rerun `just gen`,
  `just load`, then retrain. The rules read the seen phrasings from
  `templates_seen.json`, which worldgen writes (`just templates`).
- **`ood` chains word their link in a held-out way** ("the guild Orlan Teste
  is a member of"): rules that spot a chain by its seen wording no longer
  pass `ood` for free. `dig`, `core_chain` and `core_read` no longer dig on
  `ood` (their Plan cue is that wording), so their `ood` chain scores drop to
  what one search finds (`dig` 77.6% → 54.3% on `ood`, chains 70% → 0%,
  as `keyword`). Train, val and test are unchanged byte for byte;
  regenerate with `just gen && just load`.
- `keyword` (and `dig`, `core_chain`) read answers by what a question asks,
  not by its opening words: a question without "which" or "who" asks a number,
  and a comparison is two names joined by "or". On `ood` they read no answer
  before (keyword 0.155 accuracy, dig 0.267); now 0.543 and 0.776, close to
  test. Scores on `ood` from before are not comparable.
- The Evaluate core reads sentence by sentence: 2 of its 4 attention heads
  per layer see only their own sentence, and a full stop becomes a fixed
  `<stop>` token. Balanced test goes from about 74% to 84%; two-hop chains
  are unchanged. Earlier checkpoints no longer load: rerun `just fit`.
- Seen phrasings are more varied: 4 or 5 sentence structures per predicate
  instead of 2 or 3 ("The height of X is N centimetres.", "In Y, X has its
  seat."), and 2 questions per predicate instead of 1. With near-identical
  phrasings a core recognised strings: 85% on test, chance on `ood`. Every
  split changes: regenerate the data and retrain; scores from before are not
  comparable.
- Each held-out sentence structure is seen for another predicate with the
  same kind of object (`tests/ood.rs` checks it). "In 1432, X was founded."
  stayed at chance on `ood` while "In ..., X ..." was only ever seen before a
  city; births now use it, and their held-out phrasing is "1432 is the year X
  was born.".
- Template words the name syllables can spell ("live", "notes", "dates")
  are excluded from names: the core would read such a name as the word.
- The `ood` split's held-out phrasings are new structures built only from
  words the seen phrasings use ("In 1432, Varnelle was founded."). Its old
  ones brought words train never shows ("altitude", "settled"), which a core
  trained from scratch cannot read: a core at 85% on test was at chance on
  `ood`. Regenerate the data (`just gen`, `just load`, `just evaluate-data`);
  `ood` scores from before are not comparable.
- `just fit` balances the Evaluate labels within each question shape and
  reports what guessing from the shape alone scores: on 20,000 worlds the
  shape gave away 62.7% of labels, the first core scored 63.8%.
- The Evaluate core reads word tokens instead of bytes: a 97-token
  vocabulary learned from the training split, the question's names as their
  rank. Read byte by byte, it stayed near chance even on one-hop questions;
  an epoch also drops from about 50 minutes to a few. Earlier checkpoints
  no longer load: rerun `just fit`.
- `log.jsonl` gives the validation accuracy per question shape
  (`val_by_shape`), which shows which kind of reasoning fails.
- A question opening with a word the training split never saw ("When" on
  `ood`) no longer has that word read as a name.
- `just fit` takes `--dropout` and `--warmup` (steps of learning-rate
  climb). A `last.pt` saved before this no longer resumes.
- A false consensus's origin is now at least 0.10 less reliable than the
  true source (was 0.05), so no decidable trap can look undecidable.

### Fixed

- `just load` with `--prune` refuses to run when no world was read, instead
  of emptying the whole database (#10).
- Evaluation with traces requires traces and predictions to cover the same
  questions and agree on the whole decision, not only on abstaining (#11).

## [0.0.1] - 2026-11-18

First tagged release: the measurement chain, the first trained core, the
first steps of the method (sort and dig) with their floors, and worlds whose
sources disagree, ready for the challenge step.

### Added

- **World generator** (Rust): fictional worlds with invented names, facts,
  sources written from templates and questions (`retrieve`, `chain`, `compare`,
  unanswerable `spot_missing`), deterministic for a given seed, split by world
  into `train`, `val`, `test` and a sealed out-of-distribution `ood` split
  (held-out phrasings and chain combinations).
- **Orchestrator**: JSONL loader into Postgres, a `search()` tool that never
  returns ground truth, and an explicit state machine (Frame → Plan → Consult →
  Evaluate → Confront → Conclude) that writes a trace for every question.
- **Baseline models**: `random`, `always_abstain`, `keyword`; `rule_evaluate`, a
  hand-written Evaluate that knows the seen templates; `dig`, which searches
  again from what its first reading revealed.
- **Evaluation**: accuracy, **justified accuracy** (right *and* citing every
  source it needs), citation precision and recall, abstention precision, recall
  and F1, by primitive and hops; reflection scores from the traces (reading,
  rounds, early/late conclusions, grounded abstention, `dug`).
- **Evaluate core** (PyTorch, CPU): a tiny byte-level transformer trained from
  scratch to judge whether what was read is enough, with a memory guard that
  keeps training from freezing the host machine.
- **Sources that disagree** (JSONL v1, first part of the challenge step):
  traps around the fact a question asks about (contradiction, false consensus
  with attributed copies, undecidable), with their ground truth in the data and
  the database. See `docs/design-challenge.md`.
- CI on GitHub Actions running every pre-push check and the chain end to end,
  with a read-only token.
- Contributor docs: CONTRIBUTING.md, code of conduct (Contributor Covenant
  2.1), security policy, issue and pull request templates, and a test that
  keeps every package on the released version.

[Unreleased]: https://github.com/antikkorps/docta-nm/compare/v0.0.2...HEAD
[0.0.2]: https://github.com/antikkorps/docta-nm/compare/v0.0.1...v0.0.2
[0.0.1]: https://github.com/antikkorps/docta-nm/releases/tag/v0.0.1
