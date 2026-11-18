# Docta-nm

Not super intelligence: **super reflection**. Docta-nm is a frugal AI, built from scratch, whose strength is structuring its reasoning to reach a goal, not knowing things.
Its core holds **no facts**, only a method: it frames the question, plans what to look for, searches sources, knows when it has enough, and answers citing them, or abstains and says why. It is meant as a thinking partner rather than a fast answerer: it raises what a request leaves unsaid, you can step into its reflection while it runs, and it refines its reasoning from the exchange, as one would think with another person.
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

Answers are not enough: the same run scores the **reflection** from the traces.

```
reflection        read R    read P    rounds     concl     early      late  searched  grounded  missing
-------------------------------------------------------------------------------------------------------
random             0.654     0.237     1.500     0.480     0.493     0.078     0.911     0.441        0
always_abstain     0.000         —     0.000     0.000         —         —     0.000     0.000        0
keyword            0.864     0.291     1.000     0.946     0.193     0.000     1.000     0.078        0
```

- `read R` / `read P`: share of the needed sources it read / of what it read that was needed (answerable questions).
- `rounds`: search rounds per question (a plan with no query does not count). `concl`: share of answerable questions it concluded.
- `early` / `late`: among those conclusions, concluded before reading what it needed / kept searching after having it.
- `searched`: share of all its abstentions that came after at least one search.
- `grounded`: share of unanswerable questions it abstained on **after reading an excerpt about the entity asked about**. A search that never touches the subject does not count.

`always_abstain` reaches 0.50 abstention F1 with a grounded rate of 0: right outcome, no reflection.

`just up && just test` runs every check (Rust and Python, database included); run it before pushing.

## How it works

- **`worldgen/`** (Rust) generates fictional worlds: invented names, entities and facts, sources written from English templates, and questions (`retrieve`, `chain`, `compare`, plus unanswerable `spot_missing`). Every entity has one fact no source states, and unanswerable questions take every shape, so neither mention counts nor wording give abstention away. Output is byte-identical for the same seed. No LLM is used anywhere.
- **`orchestrator/`** (Python) loads the JSONL into Postgres, exposes `search(world_id, query, k)` (full-text search that never returns ground truth), and runs each question through an explicit state machine: Frame → Plan → Consult → Evaluate (back to Plan if needed) → Confront → Conclude, with exits `ill_posed_question`, `not_found` and `unresolved_contradiction`. Models only fill states; every run writes a trace to `data/traces/`.
- **`eval/`** (Python) scores predictions: answer accuracy, citation precision and recall, and abstention precision, recall and F1, overall and by primitive and hops.

Design notes and the phase 0 brief: [`docs/`](docs/).

## License

[Apache-2.0](LICENSE)
