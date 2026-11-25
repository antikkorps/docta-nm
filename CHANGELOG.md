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

- **The method, step by step** (`just steps`): one row per step (search,
  judge, read, dig, confront, conclude, human) and per measure, on test, `val`
  and `ood` side by side, with the change since the last release's snapshot
  in `scores/`, kept in the repository. `just human` now also saves its scores
  (`predictions/<split>/human/scores.json`).

### Changed

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
