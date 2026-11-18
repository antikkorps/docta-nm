# Contributing to Docta-nm

Thank you for your interest. Docta-nm builds an AI that answers because it
checked, not because it knows: its core holds no facts and follows a method
(collect, sort, challenge, dig, answer), which map onto the orchestrator's
states (Plan + Consult, Evaluate, Confront, the Evaluate → Plan loop-back,
Conclude). Read the [README](README.md) first;
design notes live in [`docs/`](docs/).

By taking part you agree to follow the [code of conduct](CODE_OF_CONDUCT.md).
Security problems go through [SECURITY.md](SECURITY.md), never a public issue.

## Set up

You need [Docker](https://docs.docker.com/get-docker/),
[Rust](https://rustup.rs/) (the pinned toolchain installs itself),
[uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
cp .env.example .env   # set one password (openssl rand -hex 32) in POSTGRES_PASSWORD and DATABASE_URL
uv sync
just up && just gen && just load && just baseline
```

## Branches and pull requests

- `main` holds releases; `dev` is the integration branch. Both are protected.
- Start every topic from an up-to-date `dev` on its own branch, and open the
  pull request **into `dev`**.
- Without write access, work from a fork:

  ```bash
  git clone https://github.com/<you>/docta-nm.git && cd docta-nm
  git remote add upstream https://github.com/antikkorps/docta-nm.git
  git fetch upstream dev
  git switch -c my-topic upstream/dev
  # …commit, push to your fork, then open a pull request into upstream `dev`
  ```
- One topic per pull request, small commits, one subject per commit.
- `dev` is merged into `main` when a set of work is stable, then tagged.

## How we work

- **Tests first.** Write the failing test, see it fail, then make it pass.
- **Before every push**: `just up && just test` (Rust fmt, clippy and tests,
  ruff, pytest with the database). Never skip hooks with `--no-verify`.
- **Small units.** Functions under 30 lines, files under about 200 lines, split
  by responsibility. Factor out on the third repetition, not before.
- **English everywhere** in code, comments, data and docs. Comments say *why*.
- **No new dependency** without saying which one and why in the pull request.
- **Changelog.** Add a line under `Unreleased` in [CHANGELOG.md](CHANGELOG.md)
  for anything a user or contributor would notice.
- **Scores.** If a change can move results, paste the `just baseline` (and
  `just baseline ood`) tables before and after in the pull request.

## Rules that are the point of the project

- **No LLM, external API or pretrained model** anywhere in the chain, data
  generation included. Names, sources and questions come from syllables and
  templates.
- **Ground truth never reaches a model.** `search()` returns only `source_id`,
  `title`, `body` and `reliability`.
- **The loop lives in the orchestrator**; a model only fills each state.
  Anything computable goes through deterministic code.
- **Determinism**: same seed and options give byte-identical data.
- **The `ood` split stays sealed**: its phrasings never appear in other splits.

The full list is in [CLAUDE.md](CLAUDE.md), which is also what AI coding
assistants working on the repository follow.

## Releasing (maintainers)

1. On a branch from `dev`: move the `Unreleased` lines under a new version in
   `CHANGELOG.md`, and set that version in `worldgen/Cargo.toml` and in
   `orchestrator/`, `eval/` and `train/` `pyproject.toml` (the root
   `pyproject.toml` is a virtual workspace with no version;
   `tests/test_versions.py` checks the four agree). Then run `uv lock` and
   `cargo check` so `uv.lock` and `Cargo.lock` follow, and commit them too:
   CI installs with `--locked`.
2. Pull request into `dev`, then a pull request from `dev` into `main`.
3. Tag `main` (`vX.Y.Z`) and publish a GitHub Release with the changelog
   section and the `just baseline` scores on `test` and `ood`.
