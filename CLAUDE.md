# Docta-nm

## Project

Research project: build from scratch a frugal AI (low RAM, low disk) whose core
holds **no facts**. The core carries a method: it knows it doesn't know, searches
sources, weighs them, and concludes citing them. Facts live in sources, never in
weights.

Current phase: **phase 0**, no training. Build the measurement chain around the
future model:

```
worldgen (Rust) → JSONL → Postgres → orchestrator (fake models) → eval → score table
```

Done when, on a clean machine with Docker, Rust, uv and just:
`just up && just gen && just load && just baseline` prints scores for the three
fake models.

- Phase 0 brief (scope, schema, JSONL v0, required tests): @docs/BRIEF-phase0.md
- Living spec (vision, loop, decisions): https://claude.ai/artifact/M8Aj7PFp75ooF3dhXfaR4R
- The brief is written in French; the code base is **English only** (see rules).

Stack: Rust 1.95 (edition 2024, pinned in `rust-toolchain.toml`) for `worldgen/`; Python 3.12 managed by a uv
workspace for `orchestrator/` and `eval/`; Postgres `pgvector/pgvector:pg17` in
Docker; psycopg 3; ruff; pytest; GitHub Actions (public GitHub repo).

## Commands

```bash
just up          # start Postgres (docker compose, waits for healthcheck)
just down        # stop it
just check-db    # smoke check: vector extension + 5 tables exist
just reset-db    # drop the volume, re-apply db/schema.sql (needed after any schema change)
just gen         # cargo run worldgen → data/{train,val,test}.jsonl
just load        # load data/*.jsonl into Postgres, mirroring it (--prune)
just baseline    # run the 3 fake models + eval, print score table
just test        # every check below, database tests included

cargo fmt --check && cargo clippy -- -D warnings && cargo test   # Rust
uv sync                                                          # Python deps
uv run ruff check && uv run ruff format --check && uv run pytest # Python
```

## Before pushing

```bash
just up && just test
```

`just test` runs cargo fmt/clippy/test, ruff check/format and pytest with `.env`
loaded. A bare `uv run pytest` skips the database tests (no `DATABASE_URL`).

## Workflow

- Plan first, then TDD: write the failing test and show the RED output before
  implementing; re-run and show the GREEN output after.
- One function, one responsibility. Functions under 30 lines, files under ~200
  lines; split by responsibility, never arbitrarily.
- DRY with judgement: factor out on the third repetition; no abstraction with a
  single caller. Flag DRY violations.
- Use current best practices for the versions already installed. No deprecated
  APIs. Never add a dependency without saying so and why.
- Comments in English, explaining why, not what.
- At the end of each feature, delegate a code review to a subagent that
  receives only the diff and the spec.
- Before every push: run the "Before pushing" command from the repo's
  CLAUDE.md; do not push if it fails; never use `--no-verify`; paste the test
  output in the summary.

Plus, for this repo:

- Small commits, one topic per commit.
- Follow the work order in the brief (infra → worldgen → loader → eval →
  orchestrator → baseline/CI/README).

## Code structure

```
justfile             # task runner (replaces the brief's Makefile); just is a prerequisite
docker-compose.yml   # db service, 127.0.0.1:5432 only, named volume, pg_isready
db/schema.sql        # mounted in docker-entrypoint-initdb.d; schema from the brief
worldgen/            # Rust crate docta-worldgen (clap, serde, rand with the chacha feature, pinned)
orchestrator/        # Python pkg docta_orchestrator: loader, state machine, search, fake models
eval/                # Python pkg docta_eval: metrics, console table, JSON export
data/                # generated JSONL and traces/ — gitignored, never edit by hand
predictions/         # one JSONL per model + scores.json from `just baseline` — gitignored
docs/                # brief and design notes
.github/workflows/   # CI: same checks as "Before pushing"
```

`train/` (PyTorch) comes in a later phase; do not create it now.

## Project-specific rules

**Hard constraints (the point of the project)**
- Never introduce an LLM, external API or pretrained model anywhere in the chain,
  including data generation. Names, source texts and questions come from
  syllable assembly and templates only.
- Ground truth (`facts`, `fact_ids`, `required_*`, `answer`) is never exposed to
  a model. `search()` returns only `source_id`, `title`, `body`, `reliability`.
- The loop lives in the orchestrator (explicit state machine); a model only
  fills each state. Anything computable (arithmetic, comparison) goes through
  deterministic code, not the model.
- Out of scope in phase 0: training, GPU, vector search (pgvector installed but
  unused), noisy or contradictory sources.

**Data**
- JSONL is canonical; Postgres is a projection reloaded from it. Loading is
  idempotent: reloading a world replaces it.
- Every JSONL line carries `"schema":"v0"`. Changing the format means a new
  schema version, not a silent edit.
- Determinism: same seed + same options ⇒ byte-identical output. Per-world seeds
  derive from the global seed; use `ChaCha8Rng` (rand `chacha` feature; OS RNG features are off), never thread/OS RNG; iterate
  ordered collections (no `HashMap` iteration in output paths).
- Train/val/test split is by world, never by question. It derives from the
  world seed, so adding worlds never moves existing ones.
- No shortcut to abstention: every entity has exactly one fact no source
  states (so being under-documented never singles out a subject), and
  unanswerable questions take every shape (retrieve, chain, compare) in the
  same proportion. `worldgen/tests/quality.rs` guards both; keep it green
  when changing sources or questions.
- Unanswerable questions all export `primitive: "spot_missing"`; their shape
  shows only through `hops`.
- FTS uses the `simple` config on purpose: names are invented, no stemming.

**English everywhere**
The brief's French identifiers and texts are translated. Use these names:
- primitives: `retrieve` (1 hop), `chain` (2 hops), `compare`, `spot_missing`
- states: `Frame → Plan → Consult → Evaluate → (back to Plan) → Confront → Conclude`
- emergency exits: `ill_posed_question`, `not_found`, `unresolved_contradiction`
- entity kinds and predicates in English snake_case (`city`, `person`,
  `founded_in`, `lives_in`); source/question templates written in English.
- Fake models: `RandomModel`, `AlwaysAbstainModel`, `KeywordModel`.

**Secrets and infra**
- Credentials only in `.env` (gitignored); keep `.env.example` in sync.
  `DATABASE_URL` is read from the environment.
- Postgres port bound to `127.0.0.1` only.

**When unsure**
On any structural choice not settled by the brief or this file, ask Franck
instead of deciding alone.
