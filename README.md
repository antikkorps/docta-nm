# Docta-nm

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-fvienot-FFDD00?logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/fvienot)

**An AI that answers because it checked, not because it knows.**

Most AI assistants answer from what they memorised during training, and sound just as sure when they are wrong. Docta-nm takes the opposite path: its core memorises **no facts**. It carries a method instead, the one a careful researcher follows:

1. **Collect**: search the sources for what the question needs.
2. **Sort**: keep what is relevant, and know when there is enough.
3. **Challenge**: weigh sources against each other and spot contradictions.
4. **Dig**: follow up on what was just learned, until nothing is missing.
5. **Answer**: conclude citing the sources, or say plainly "I could not find it" and why.

Every answer can be traced back to the sources that establish it. Since knowledge lives in the sources and not in the model, the core stays small enough to run on a modest machine.

It is meant as a thinking partner rather than a fast answerer: it raises what a request leaves unsaid, and you can step into its reasoning while it runs.

### How it is tested

To be sure the core reads instead of remembering, it is trained and tested on **invented worlds**: fictional cities, people and guilds with made-up names and facts, described by generated sources. Nothing can be known in advance; the only way to answer is to search and read. A separate test set uses phrasings never seen in training, to check the core learned the method and not the wording.

### Where it stands

- **Done**: the full measurement chain (world generator, sources in Postgres, the reasoning loop, scores on both the answers and the reasoning traces), with simple baseline models to beat.
- **In progress**: the first tiny core, trained from scratch on one step of the method: judging whether what it has read is enough.
- **Next**: digging (a second search driven by what the first one revealed), then challenging (sources that disagree).

```
fictional worlds (Rust) → JSONL → Postgres → reasoning loop + models → evaluation → scores
```

## Quick start

Requirements: [Docker](https://docs.docker.com/get-docker/), [Rust](https://rustup.rs/) (the pinned toolchain installs itself), [uv](https://docs.astral.sh/uv/) and [just](https://just.systems/).

```bash
cp .env.example .env   # then set a password (openssl rand -hex 32) in both places
uv sync
just up                # Postgres with pgvector, on 127.0.0.1:5442
just gen               # 1000 fictional worlds → data/{train,val,test,ood}.jsonl
just load              # data/*.jsonl → Postgres
just baseline          # run the fake models and the Evaluate rule on test, print their scores
just baseline ood      # the same on ood: phrasings and chains never seen in training
```

`just baseline` prints, among other tables:

```
model                acc  just acc    cite P    cite R    abst P    abst R   abst F1  bad abst  missing
-------------------------------------------------------------------------------------------------------
random             0.007     0.002     0.181     0.116     0.333     0.530     0.409     0.532        0
always_abstain     0.000     0.000         —     0.000     0.333     1.000     0.500     1.000        0
keyword            0.553     0.499     0.451     0.506     0.443     0.086     0.144     0.054        0
rule_evaluate      0.536     0.499     0.774     0.487     0.676     1.000     0.806     0.240        0
```

`keyword` answers easy questions but almost never abstains; `always_abstain` does the opposite. The trained core has to beat **0.50 abstention F1** while still answering.

`just acc` counts a right answer only when it cites every source the question needs: the answer must come from evidence that establishes it, not from knowing it. The gap to `acc` is lucky guesses.

`rule_evaluate` is keyword with a hand-written Evaluate that knows the seen templates: it says "enough" only when every needed fact was read. On test it is close to an oracle; on `ood` it understands no phrasing and falls back to keyword, so `ood` is where a trained core has to beat it.

`dig` is keyword plus a second search: when a question refers to something indirectly ("the city where X lives"), it follows the name the first reading revealed ("X lives in Varnelle") and reads the answer there, citing both sources. It answers 56% of 2-hop questions on test, where keyword answers none.

Answers are not enough: the same run scores the **reflection** from the traces.

```
reflection        read R    read P    rounds     concl     early      late  searched  grounded  missing
-------------------------------------------------------------------------------------------------------
random             0.653     0.235     1.557     0.484     0.486     0.068     0.903     0.376        0
always_abstain     0.000         —     0.000     0.000         —         —     0.000     0.000        0
keyword            0.855     0.285     1.000     0.941     0.201     0.000     1.000     0.075        0
```

- `read R` / `read P`: share of the needed sources it read / of what it read that was needed (answerable questions).
- `rounds`: search rounds per question (a plan with no query does not count). `concl`: share of answerable questions it concluded.
- `early` / `late`: among those conclusions, concluded before reading what it needed / kept searching after having it.
- `searched`: share of all its abstentions that came after at least one search.
- `grounded`: share of unanswerable questions it abstained on **after reading an excerpt about the entity asked about**. A search that never touches the subject does not count.
- `dug`: among answerable questions whose first search missed a needed source, share where later searches found it: **digging** from what was read.

`always_abstain` reaches 0.50 abstention F1 with a grounded rate of 0: right outcome, no reflection.

**Generalisation.** The `ood` split is written with sentence and question phrasings, and chain combinations, that train, val and test never contain. On it, `keyword` drops from 0.52 to 0.18 accuracy: it recognised templates rather than reading. The trained core has to hold up on `ood` too, or it has learned the templates, not the method.

`just up && just test` runs every check (Rust and Python, database included); run it before pushing.

## How it works

- **`worldgen/`** (Rust) generates fictional worlds: invented names, entities and facts, sources written from English templates, and questions (`retrieve`, `chain`, `compare`, plus unanswerable `spot_missing`). Every entity has one fact no source states, and unanswerable questions take every shape, so neither mention counts nor wording give abstention away. Output is byte-identical for the same seed. No LLM is used anywhere.
- **`orchestrator/`** (Python) loads the JSONL into Postgres, exposes `search(world_id, query, k)` (full-text search that never returns ground truth), and runs each question through an explicit state machine: Frame → Plan → Consult → Evaluate (back to Plan if needed) → Confront → Conclude, with exits `ill_posed_question`, `not_found` and `unresolved_contradiction`. Models only fill states; every run writes a trace to `data/traces/`.
- **`eval/`** (Python) scores predictions: answer accuracy (raw and justified by its citations), citation precision and recall, and abstention precision, recall and F1, overall and by primitive and hops.

Design notes and the phase 0 brief: [`docs/`](docs/).

## License

[Apache-2.0](LICENSE)
