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

### The name

**Docta** comes from *docta ignorantia*, the "learned ignorance" of Nicholas of Cusa (1440): true wisdom begins with knowing what one does not know. That is the core's starting point: it knows it does not know, so it searches, weighs, and concludes only from what it found. **nm** stands for *neural model*.

### How it is tested

To be sure the core reads instead of remembering, it is trained and tested on **invented worlds**: fictional cities, people and guilds with made-up names and facts, described by generated sources. Nothing can be known in advance; the only way to answer is to search and read. A separate test set uses phrasings never seen in training, to check the core learned the method and not the wording.

### Where it stands

- **Done**: the full measurement chain (world generator, sources in Postgres, the reasoning loop, scores on both the answers and the reasoning traces), with simple baseline models to beat; digging, with the `dig` baseline; and sources that disagree, scored by the trap table, with the `majority` and `challenge` baselines for the challenge step.
- **In progress**: the first tiny core, trained from scratch on one step of the method: judging whether what it has read is enough. The training chain is in place (see [Training the Evaluate core](#training-the-evaluate-core)); a trained Confront for the challenge step is still to come.
- **Next**: beat the baselines with the trained core, on `ood` as well as on test.

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

**Traps.** Some questions surround their fact with sources that disagree: a less reliable source stating a false value (`contradiction`), copies repeating one false origin (`false_consensus`), or two equally reliable sources that cannot be settled (`undecidable`, where the right move is the exit `unresolved_contradiction`). A `model · trap` table scores them: `acc` and `just acc` on the decidable ones, `fooled` (answered the false value, or settled an undecidable one), and the precision and recall of the `unresolved_contradiction` exit. Two hand-written Confronts show what the step is about. `majority` gives each source one vote: copies of one false origin outvote the truth, and it is fooled by every false consensus on test. `challenge` counts a source that credits another ("According to …") together with that origin, weighs independent sources by reliability, and exits `unresolved_contradiction` when two are too close to call: never fooled by a false consensus, and it takes the exit on every undecidable question, always rightly. Both read the seen templates, so on `ood` they fall back to keyword: a trained Confront has to beat them there.

**Generalisation.** The `ood` split is written with sentence and question phrasings, and chain combinations, that train, val and test never contain. Its phrasings are new structures built only from words the seen ones use ("In 1432, Varnelle was founded.", "Which of Varnelle or Brask lies higher?"): the core learns its words from train only, so an unseen word would test vocabulary rather than the method, and held-out words such as "altitude" left a core at chance on `ood`. The trained core has to hold up on `ood` too, or it has learned the templates, not the method.

`just up && just test` runs every check (Rust and Python, database included); run it before pushing.

### Training the Evaluate core

The `train/` package (PyTorch, CPU) trains the Evaluate core and runs it. After `just up`, `just gen` and `just load`:

```bash
just evaluate-data     # data/{train,val,test,ood}.jsonl → data/evaluate: question + excerpts read → enough or not
just fit               # train the core on data/evaluate → models/core_evaluate (model.pt, log.jsonl)
just core              # run the trained core on test and score it next to the baselines
just core ood          # the same on ood
```

- `just evaluate-data` builds each example from the question and the excerpts the real `search()` returns; the label is "enough" only when the question has an answer and every source it needs was read.
- `just fit` passes its arguments to `docta_train.fit`: `--epochs`, `--batch`, `--lr`, `--seed`, `--max-train`, `--max-val`, `--max-minutes`, `--max-gb`, `--threads`, `--dropout` and `--warmup` (for example `just fit --epochs 5`). The memory and thread caps keep a training run from freezing the machine it shares; the address-space cap is applied on Linux only.
- `just fit` balances labels within each question shape (the question with its names and numbers hidden), so the core cannot score by recognising the template instead of reading the excerpts. Its first line prints what guessing from the shape alone scores on val, before and after balancing: the core reads only above that line. `--no-balance` keeps every example.
- Each epoch also writes `models/core_evaluate/last.pt`. `just fit --resume --epochs 6` adds six epochs to that run, exactly as if it had never stopped (same shuffling, same optimiser state); the run keeps its learning rate unless `--lr` is given. A resume on other data or another core is refused.
- `--dropout` and `--warmup` (steps of learning-rate climb) regularise the run. On 20,000 worlds, `just fit --epochs 22 --dropout 0.1 --warmup 1000 --lr 1e-3` gave the best validation loss so far (0.485 against 0.510 without, 74.8% accuracy against 74.2%); without warmup, `--lr 1e-3` locks the core on one constant answer. A resumed run keeps its dropout and its place in the warmup.
- The core reads word tokens. The vocabulary is learned from the training split only (words seen in at least 20 worlds: the templates' words, never an invented name) and saved with the checkpoint. The question's names become their rank (`<name_1>`…`<name_4>`), other names `<name>`, numbers `<num>`, so the core sees where each name recurs without reading its spelling. Each line of `log.jsonl` also gives the validation accuracy per question shape (`val_by_shape`), which shows which kind of reasoning fails. A checkpoint from before word tokens no longer loads: rerun `just fit`.
- Half the attention heads of each layer (`local_heads`, 2 of 4) read only within one sentence: the question, each title and each body sentence, as code numbers them from `<src>`, `<body>` and the `<stop>` token a full stop becomes. The other heads read everything. Without them the core saw a property and a name somewhere in the excerpts, not stated together (74% on balanced val); with them, balanced test reaches about 84%: one-hop questions about 94%, comparisons 100%, two-hop chains still 55–60%. On `ood` the core scores about 60–65% and chains stay at chance. Over three seeds, one run in three stalls near 70% for its 10 epochs: compare settings over several seeds, never on one run. A checkpoint from before `<stop>` no longer loads: rerun `just fit`.
- `just core` writes `core_evaluate` predictions next to the baselines in `predictions/<split>` and adds it to the same score tables.

## How it works

- **`worldgen/`** (Rust) generates fictional worlds: invented names, entities and facts, sources written from English templates, and questions (`retrieve`, `chain`, `compare`, plus unanswerable `spot_missing`). Every entity has one fact no source states, and unanswerable questions take every shape, so neither mention counts nor wording give abstention away. Output is byte-identical for the same seed. No LLM is used anywhere.
- **`orchestrator/`** (Python) loads the JSONL into Postgres, exposes `search(world_id, query, k)` (full-text search that never returns ground truth), and runs each question through an explicit state machine: Frame → Plan → Consult → Evaluate (back to Plan if needed) → Confront → Conclude, with exits `ill_posed_question`, `not_found` and `unresolved_contradiction`. Models only fill states; every run writes a trace to `data/traces/`.
- **`eval/`** (Python) scores predictions: answer accuracy (raw and justified by its citations), citation precision and recall, and abstention precision, recall and F1, overall and by primitive and hops.

Design notes and the phase 0 brief: [`docs/`](docs/).

## Contributing

There is a lot to build, and help is welcome: see [CONTRIBUTING.md](CONTRIBUTING.md)
for the setup, the branch flow and the project rules, and [CHANGELOG.md](CHANGELOG.md)
for what changed in each version.

## License

[Apache-2.0](LICENSE)
