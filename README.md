# Docta-nm

A frugal AI built from scratch whose core holds **no facts**, only a method.
It knows that it does not know: it searches sources, weighs them, and answers citing them, or abstains.
Facts live in the sources, never in the weights, so the core stays small in RAM and on disk.

This repository is at **phase 0**: no model is trained yet. Phase 0 builds what the first trained core will be measured with:

```
fictional worlds (Rust) → JSONL → Postgres → orchestrator + fake models → evaluation → scores
```

The first milestone is a tiny core, trained from scratch, that **abstains correctly** on questions the sources cannot answer and beats the fake models below.

## Quick start

Requirements: [Docker](https://docs.docker.com/get-docker/), [Rust](https://rustup.rs/) (the pinned toolchain installs itself), [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
cp .env.example .env   # then set a password (openssl rand -hex 32) in both places
uv sync
just up                # Postgres with pgvector, on 127.0.0.1:5442
just gen               # 1000 fictional worlds → data/{train,val,test}.jsonl
just load              # data/*.jsonl → Postgres
just baseline          # run the three fake models on the test split, print their scores
```

`just baseline` prints, among other tables:

```
model                acc    cite P    cite R    abst P    abst R   abst F1  bad abst  missing
---------------------------------------------------------------------------------------------
random             0.005     0.194     0.119     0.344     0.546     0.422     0.520        0
always_abstain     0.000         —     0.000     0.333     1.000     0.500     1.000        0
keyword            0.547     0.478     0.537     0.431     0.082     0.137     0.054        0
```

`keyword` answers easy questions but almost never abstains; `always_abstain` does the opposite. The trained core has to beat **0.50 abstention F1** while still answering.

`just up && just test` runs every check (Rust and Python, database included); run it before pushing.

## How it works

- **`worldgen/`** (Rust) generates fictional worlds: invented names, entities and facts, sources written from English templates, and questions (`retrieve`, `chain`, `compare`, plus unanswerable `spot_missing`). Every entity has one fact no source states, and unanswerable questions take every shape, so neither mention counts nor wording give abstention away. Output is byte-identical for the same seed. No LLM is used anywhere.
- **`orchestrator/`** (Python) loads the JSONL into Postgres, exposes `search(world_id, query, k)` (full-text search that never returns ground truth), and runs each question through an explicit state machine: Frame → Plan → Consult → Evaluate (back to Plan if needed) → Confront → Conclude, with exits `ill_posed_question`, `not_found` and `unresolved_contradiction`. Models only fill states; every run writes a trace to `data/traces/`.
- **`eval/`** (Python) scores predictions: answer accuracy, citation precision and recall, and abstention precision, recall and F1, overall and by primitive and hops.

Design notes and the phase 0 brief: [`docs/`](docs/).

## License

[Apache-2.0](LICENSE)
