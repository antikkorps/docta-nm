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

- **Trap scores**: a `model · trap` table and a `traps` entry in the JSON
  scores: accuracy and justified accuracy on decidable traps, how often a
  model was fooled, and the precision and recall of the
  `unresolved_contradiction` exit.
- **Confront baselines**: `majority` (one vote per source, fooled by false
  consensus) and `challenge` (copies count with their origin, independent
  sources weighed by reliability, `unresolved_contradiction` when too close).

### Changed

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
