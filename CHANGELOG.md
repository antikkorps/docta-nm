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

[Unreleased]: https://github.com/antikkorps/docta-nm/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/antikkorps/docta-nm/releases/tag/v0.0.1
