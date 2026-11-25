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

- **Done**: the measurement chain end to end (a generator of fictional worlds, their sources in Postgres, the reasoning loop, and scores for the answers as well as the reasoning traces) with simple baselines to beat; the digging step, with `dig` as its baseline; and sources that disagree, scored by the trap table, with `majority` and `challenge` as baselines.
- **In progress**: tiny cores trained from scratch, one step of the method each. The Evaluate core judges whether what was read is enough; the Read core points at the answer in it, and code copies and cites it (see [Evaluate core: training and running](#evaluate-core-training-and-running) and [Training the Read core](#training-the-read-core)). On test, `core_read` beats every baseline (83.0% exact, all justified, with the grammar's data and the bridge searched by its name); on `ood` it stays at 13–32% depending on the Evaluate run, because the Evaluate core wrongly abstains on 60–80% of answerable questions phrased in held-out ways. Since `ood` holds out its chain link wording too, no rule digs there any more: `dig` falls from 77.6% to `keyword`'s 54.3% on `ood`.
- **Next**: make `ood` honest and then pass it: a chain link phrasing held out for `ood`, many more sentence structures per predicate, then a learned Plan, a trained Confront, and a measure of how the core resumes when a human corrects it mid-way.

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

`rule_evaluate` is keyword with a hand-written Evaluate that knows the seen templates (worldgen exports them to `templates_seen.json`; `just templates` refreshes it after a template change): it says "enough" only when every needed fact was read. On test it is close to an oracle; on `ood` it understands no phrasing and falls back to keyword, so `ood` is where a trained core has to beat it.

`dig` is keyword plus a second search: when a question refers to something indirectly ("the city where X lives"), it searches for the name the first reading revealed ("X lives in Varnelle"), by that name alone, and reads the answer there, citing both sources. The question's other words ("where", "city", "lives") would bring back sources about anything near the bridge: on test, the name alone finds the bridge's value in the top five for 98% of the chains `core_read` missed, the bridge with the question's other words for 33%. It answers 57% of 2-hop questions on test (grammar data), where keyword answers none.

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

**Generalisation.** The `ood` split is written with sentence, question and chain link phrasings, and chain combinations, that train, val and test never contain ("the guild Orlan Teste is a member of" where the others say "the guild that Orlan Teste belongs to"), so rules that spot a chain by its seen wording do not pass `ood` for free. Sentences come from a small grammar: each relation clause takes eight frames ("Records show that ...", "..., the records say.") with or without a parenthetical after its subject ("Varnelle, it is recorded, was founded in 1432."), 30–48 structures per predicate, so the subject and the value never sit at fixed places. `ood` holds out the held-out clause in every form and one (clause, form) pair in four: every piece is seen in train, never all their pairings. Its phrasings are new structures built only from words the seen ones use ("In 1432, Varnelle was founded.", "Which of Varnelle or Brask lies higher?"): the core learns its words from train only, so an unseen word would test vocabulary rather than the method, and held-out words such as "altitude" left a core at chance on `ood`. The trained core has to hold up on `ood` too, or it has learned the templates, not the method.

**Choosing settings without opening `ood`.** `val` is written in phrasings of its own too: one sentence clause per predicate in every form, and one question, comparison and chain link phrasing each ("How many inhabitants does Varnelle have?", "Of Varnelle or Brask, which lies higher?", "the city that is home to Orlan Teste"), never in train, test or `ood`, and built only from seen words. So `val` measures generalisation and epochs or settings can be picked there while `ood` stays sealed; `test` keeps measuring the seen phrasings. Train, test and `ood` are written byte for byte as before; the rules never see `val`'s phrasings (`tests/val_held.rs`).

`just up && just test` runs every check (Rust and Python, database included); run it before pushing.

### Evaluate core: training and running

The Evaluate core is trained and run by the `train/` package (PyTorch, on CPU). Once `just up`, `just gen` and `just load` have run:

```bash
just evaluate-data     # builds data/evaluate from data/{train,val,test,ood}.jsonl: a question, the excerpts read, labelled enough or not
just fit               # trains the core on data/evaluate, writes models/core_evaluate (model.pt, log.jsonl)
just core              # scores the trained core on test beside the baselines
just core ood          # likewise on ood
just human             # each chain with no hint, the right bridge and a wrong one said mid-way
```

- Each example `just evaluate-data` writes pairs a question with what the real `search()` brought back for it, and is labelled "enough" only if that question can be answered and all the sources it requires were among the excerpts. Each question not yet enough also gets a second example "after the dig": the excerpts plus what a search for the next name brings (a chain's bridge, from ground truth, once its link was read; otherwise the first name a sentence about the subject brings in). Comparisons never dig. With it, one Evaluate seed (16 epochs) reaches AUC 0.953 on test and 0.818 on `ood` (0.635–0.753 before); the other seed tried stayed on shape alone for 9 epochs and was stopped.
- Arguments given to `just fit` go to `docta_train.fit`: `--epochs`, `--batch`, `--lr`, `--seed`, `--max-train`, `--max-val`, `--max-minutes`, `--max-gb`, `--threads`, `--dropout` and `--warmup` (for example `just fit --epochs 5`). Capping memory and threads lets a run share the machine without freezing it; the address-space limit only takes effect on Linux.
- `just fit` balances labels within each question shape (the question with its names and numbers hidden), so the core cannot score by recognising the template instead of reading the excerpts. Its first line prints what guessing from the shape alone scores on val, before and after balancing: the core reads only above that line. `--no-balance` keeps every example.
- Each epoch also writes `models/core_evaluate/last.pt`. `just fit --resume --epochs 6` adds six epochs to that run, exactly as if it had never stopped (same shuffling, same optimiser state); the run keeps its learning rate unless `--lr` is given. A resume on other data or another core is refused.
- `--dropout` and `--warmup` (steps of learning-rate climb) regularise the run. They default to `--dropout 0.1 --warmup 4000 --lr 1e-3`: over three seeds, these got every run off the shape-alone plateau within 2 to 3 epochs (val accuracy 68.8–76.7% after 6, on `val`'s held phrasings), where `--warmup 1000` left two seeds in four on it and `--lr 3e-4` overfitted. Without warmup, `--lr 1e-3` locks the core on one constant answer. A resumed run keeps its dropout and its place in the warmup.
- The core reads word tokens. The vocabulary is learned from the training split only (words seen in at least 20 worlds: the templates' words, never an invented name) and saved with the checkpoint. The question's names become their rank (`<name_1>`…`<name_4>`), other names `<name>`, numbers `<num>`, so the core sees where each name recurs without reading its spelling. Each line of `log.jsonl` also gives the validation accuracy per question shape (`val_by_shape`), which shows which kind of reasoning fails. A checkpoint from before word tokens no longer loads: rerun `just fit`.
- Half the attention heads of each layer (`local_heads`, 2 of 4) read only within one sentence: the question, each title and each body sentence, as code numbers them from `<src>`, `<body>` and the `<stop>` token a full stop becomes. The other heads read everything. Without them the core saw a property and a name somewhere in the excerpts, not stated together (74% on balanced val); with them, balanced test reaches about 84%: one-hop questions about 94%, comparisons 100%, two-hop chains still 55–60%. On `ood` the core scores about 60–65% and chains stay at chance. Over three seeds, one run in three stalls near 70% for its 10 epochs: compare settings over several seeds, never on one run. A checkpoint from before `<stop>` no longer loads: rerun `just fit`.
- Two models come from `just core`, their predictions saved beside the baselines' in `predictions/<split>` and scored in the same tables: `core_evaluate` (keyword's plan, the core's judgement) and `core_chain` (dig's two-round plan; the orchestrator splits a chain into one-hop questions and the core judges each). For "In what year was the city where Orlan Teste lives founded?", the core judges "In which city does Orlan Teste live?" on the sentence revealing each candidate bridge only, then "In what year was Varnelle founded?" on everything read; the answer is read about a bridge that passed both. The outer wording is kept, so a held-out phrasing stays held out. On the same excerpts, the core judges chains at 90–93% balanced on test split this way, against 55–60% read whole (on `ood`, whose link wording is held out, the split no longer fires: chains are read whole). End to end on test, `core_chain` answers 57% of chains (`core_evaluate` none) and raises abstention F1 from 0.71 to 0.77. On `ood`, the Evaluate core wrongly abstains on most answerable questions (57% for one run, 71–80% for another with the same settings: a single run says little about `ood`), so its judgement of held-out phrasings is what limits it; and with the link wording held out, neither `dig` nor `core_chain` digs there.

### Training the Read core

The Read core reads the answer where Evaluate judged the excerpts enough. It points at one word of the excerpts' bodies; code copies the number or the whole name found there and cites its source, so the core never writes an answer it did not read.

```bash
just read-data         # data/evaluate + data/*.jsonl → data/read: question + excerpts judged enough → answer and its sources
just fit-read          # train it → models/core_read (model.pt, log.jsonl)
just core              # core_read joins the other cores once models/core_read/model.pt exists
```

- `just read-data` keeps the Evaluate examples labelled enough, on retrieve and compare questions. A comparison becomes one example per option, named again after the question: the core reads each value, code compares them. Answerable, trap-free chains teach what to look at next: the bridge while only the link was read, the answer once both hops were (`chain_truth.py` finds the bridge from ground truth, a target only). Traps are left out (weighing sources is Confront's step). The answer and its sources serve as training targets only, never as input.
- `just fit-read` reports `val_exact` with `val_exact_answer` and `val_exact_bridge`, one per pointer, and takes `--epochs` (2 by default), `--batch`, `--lr`, `--seed`, `--warmup`, `--dropout`, `--max-train`, `--max-minutes`, `--max-gb` and `--threads`, and keeps the last epoch. Taught retrieve and compare only, val was no guide: it read 100% exactly from epoch 1, while `ood` still moved and peaked at epoch 2 on three seeds (75%, 91%, 82%; the `keyword` rule reads 87% of the same examples). Since chains are taught too, val no longer saturates and each pointer's val score is the guide. Its weights never load as an Evaluate core, nor the reverse.
- `core_read` is `core_chain` with the Read core's answers: one hop, it reads the answer; a comparison, it reads each value and code compares; a chain, it reads the bridge on the first hop, which counts only if a sentence about the subject brings that name in (as `core_chain`'s bridges), the first hop is judged on those sentences, and the value is read on the second. An answer is a number or a name, never a word in between. End to end on test (Evaluate core `seed-v3-1`, three Read seeds) it answers 76.9% exactly (all justified) against 65.3% for `core_chain` and 68.1% for `dig` (chains 69% against 57% and 54%), with citation precision 0.91–0.94 and abstention F1 0.78–0.80. A rerun of `just fit` and `just fit-read` with the same settings gives 75.5% on test. On `ood` it stays at 13–32% depending on the Evaluate run: the Evaluate core wrongly abstains on 60–80% of answerable questions before the Read core reads anything. Retrained on the grammar's data (16 epochs, two Evaluate seeds), it answers 77.9% on test with either seed, and 83.0% (chains 86.8%) once digging searches for the bridge by its name; on `ood`, 32–37%, with chains near 0% since the split never fires on held-out link wording.
- `core_bridge` (experimental) is `core_read` with no rule on a chain's wording: the question is read whole, the Read core points at the answer or at the name to search next, code searches that name once, and Evaluate judges the whole question on all that was read. A chain cites its link and its answer. Measured once (Evaluate seed 2 trained with the dig examples, one Read seed): chains 27.9% on test, all justified, against 69.3% for `core_read` with the same checkpoints; 6.9% on `ood` (6.1% justified) against 4.6% (0.2%). An answer stands only if a source ties the question's subject to it (a sentence about the subject names the bridge, named again where the answer is said, or the answer is stated next to the subject): without that check a wrong hint made it give the wrong bridge's answer on 41% of test chains; with it, `fooled` is 0 on test and `ood`, the wrong answers turned into abstentions. The limit is the Read core: taught both targets, it points at the bridge 73% of the time on test (59% on `ood`, against 100% when taught the bridge alone), and at the answer once both hops were read 57% (21%). Six epochs instead of two do not help: val stays near 87% (chain shapes 50–75%), the bridge rises to 81–82% but the answer once both hops were read falls to 44% (19.5%), the core pointing at the bridge instead: one pointer for both targets is the limit, not training length. With these checkpoints `core_read` itself falls to 76.8% on test (83.0% before) and rises to 40.7% on `ood` (32–37%). The Read core now has two pointers on one encoding, one for the answer and one for the name to search next (`just read-data` marks which target each example teaches; data or a checkpoint from before is refused, so rerun `just read-data && just fit-read`); code digs a name only if a sentence about the subject brings it in. Measured once (same Evaluate seed 2, one Read seed, 4 epochs): the bridge head points right 100% of the time on test and 85.5% on `ood`, the answer head once both hops were read 71.4% and 23.8% (on `ood` it points at the bridge instead in 27% of cases). End to end, `core_bridge` chains rise to 47.7% on test (27.9% with one pointer) and 9.9% on `ood` (9.8% justified, 6.1% before), `fooled` stays 0; `core_read` gives 77.5% on test (chains 70.8%) and 43.9% on `ood`. Evaluate still abstains on half the answerable chains on test and two thirds on `ood`.

### A human in the reflection

The point of the method is a human–machine pair: a person can step into the reflection mid-way and the core resumes from there. `just human` (after `just core`'s training) measures it on every answerable chain, run three times: with no hint, with the right bridge ("Orlan Teste lives in Varnelle.") said before round 2, and with a wrong one of the same kind.

- The hint goes through a **human channel**: Plan hears it, so it may steer the next search; Evaluate, Confront and Conclude never do, so the answer must still stand on sources. A wrong hint can cost a search, never support a wrong answer. A model about to give up after round 1 is given that round to hear the human; a model already sure of its answer never hears it.
- Columns: accuracy with no hint, the right hint and the wrong one; `heard` (the hint reached the model) and `reoriented` (its next search went after the hinted name); `recovered` (chains failed alone that the right hint turned right); `fooled` (wrong-hint runs that answered what the wrong bridge would give).
- The simulated human knows the bridge from ground truth: the one exception to "ground truth is never a model input", at evaluation only.

On test, with the grammar's data and one Evaluate seed (before 0.0.2's retraining; [`scores/`](scores/) holds the current figures), `core_read` answers 86.8% of chains alone, 94.8% with the right hint and 72.7% with a wrong one: the right hint turns 60% of its failed chains right, and a wrong hint never makes it give the wrong bridge's answer (`fooled` 0); it costs abstentions instead. On `ood` the chain split never fires (held-out link wording), so no model has a second hop for a hint to steer.

### The method, step by step

Each step of the method is measured on its own, so a blockage shows at the step where it happens and that step can be tuned without touching the rest. `just steps` gathers one model's scores (`core_read` by default, `--model` for another) into one row per step and per measure, on test, `val` and `ood` side by side:

| Step | Main measure | Supporting measure |
|---|---|---|
| search | sources it needed that it read (`reading_recall`) | `reading_precision` |
| judge | abstention F1 | answerable questions it gave up on |
| read | accuracy on retrieve questions (comparisons are code's arithmetic) | citation precision |
| dig | justified accuracy on chains | answerable questions a later search recovered after the first missed a source |
| confront | accuracy on traps | traps that fooled it |
| conclude | justified accuracy | conclusions before reading what was needed |
| human | failed chains the right hint recovers | wrong hints it repeated |

Run `just core` (and `just human`) on each split first; a split not run is left blank. Each release saves a snapshot in [`scores/`](scores/) (`just steps --save X.Y.Z`), kept in the repository, and `just steps` prints the change since the latest one, so anyone can follow every step from one version to the next.

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

Docta-nm is dual-licensed:

- under the [GNU Affero General Public License v3.0](LICENSE) (`AGPL-3.0-only`):
  anyone may use, study, modify and share it; whoever distributes it, or lets
  users interact with a modified version over a network, must offer them its
  source under the same licence;
- under a commercial licence, for use that cannot follow the AGPL (a closed
  product or service). To ask for one, contact the maintainer through
  [GitHub](https://github.com/antikkorps).

Everything published before the switch to the AGPL (v0.0.1, and the `dev`
branch up to that change) was under Apache-2.0 and stays available under it.
Contributions are accepted under the
[Contributor License Agreement](CLA.md), which lets the project offer both
licences.
