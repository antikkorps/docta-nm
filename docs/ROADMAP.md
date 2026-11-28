# Roadmap and handoff

Where the work stands, what comes next, and how to pick it up on another
machine or in a fresh session. The living spec keeps the vision; this file
keeps the concrete state. Update it when a chantier moves, not only at
releases. Last update: 2026-11-28.

## How we work

The method we ask of the core is the one we apply to the code:

- **Diagnose before tuning.** When a score stalls, find where it fails first
  (on `val`, broken down by wording or shape), state what the fix should
  move, then test it. No setting is changed at random.
- **`ood` is sealed.** Settings are chosen on `val` only; `ood` is looked at
  once, to confirm. `val` holds phrasings of its own for that purpose.
- **Three seeds at least** before a claim enters the CHANGELOG. Seeds vary a
  lot here (Read on the M3: 13 to 24 points between seeds of one setting).
- **Screen small, confirm at full size.** A 40k-example, 3-epoch screen on 3
  seeds, then the full data on the winner.
- **Labels are checked too.** Real-text labels come with the error found by
  hand next to the scores.
- **Reviews are weighed, not obeyed.** Outside reviews and agents' reports
  are checked against the code and the rules; only what shifts a priority
  changes the plan.

The project rules themselves (no pretrained model, ground truth never an
input, determinism, branches and releases) are in `CLAUDE.md` and
`CONTRIBUTING.md`.

## Where it stands

| Chantier | State | Numbers |
|---|---|---|
| Measurement chain (phase 0) | Done | `just baseline` |
| Evaluate core (is what I read enough?) | Overlap flag confirmed (PR #86); full-loop `ood` to measure | `val` median 83.3% (3 seeds, 0.0.2: 76.7%); classifier `ood` median 82.2% (0.0.2: 77.1%) |
| Read core (where is the answer?) | Trained; default warmup 4000 | `val_exact` median 59.1% over 5 seeds (spread 8 points) |
| Language paliers | Palier 1 released (0.0.2); palier 2 begun (pronouns) | see `docs/` and the CHANGELOG |
| Real-text probe | Questions, cuts, loading, scoring, `just real`, hand check done | labels: answerable 8% wrong (20% with unsure), natural 8%, redacted 0% |
| Plan, Confront | Rules only (`keyword`, `majority`, `challenge`) | |

### Evaluate on unseen wordings (the current priority)

On `ood`, `core_read` loses to `keyword` because Evaluate says "not enough"
on answerable questions (30% wrong abstentions against keyword's 7%, with
the same reading recall, 0.847). Plan and Confront are not the cause:
Plan already brings the right sources, and Confront never sees what
Evaluate abstained on.

Diagnosis on `val` (2026-11-27): the core paired question templates with
sentence templates. "What is the guild of X Y?" was wrongly judged not
enough 86% of the time although train holds thousands of "the guild of X Y
is Z".

Fix: an overlap flag (`fit --overlap`, PR #86), DrQA-style, marking the
excerpt words the question also holds. Full data, 6 epochs, 3 seeds:
balanced `val` accuracy 80.6–84.1% (median 83.3%) against 68.8–76.7%
without. `ood`, looked at once (Evaluate classifier alone): median 82.2%
against 77.1% for the 0.0.2 core; the seed chosen beforehand on `val` (mean
accuracy per wording) wrongly abstains on 4.9% of answerable examples against
23.1%. Seeds still vary on `ood` (75.5–84.0%).

What remains (diagnosed 2026-11-28): wordings stuck near 50%, different on
each seed and flickering between epochs, all frozen on "not enough". Most
have AUC near 0.5 (the core does not separate them: guild chains, "the guild
of", two comparisons); two have AUC above 0.9 (a threshold shift), but a
global threshold chosen on `val` gains under a point and its best value
differs between seeds, so it was dropped. One guild chain wording stays near
chance on `ood` for every seed. Prediction for palier 2: if these come from
template pairing, more wordings per relation should make them disappear.

## Next, in order

1. **Close the Evaluate fix**: merge PR #86, then the full-loop `ood`
   measure (`core_read` against `keyword`, the original bottleneck) with the
   seed 3 checkpoint, on the 0.0.2 worlds loaded in the database.
2. **First real-text scores**: label fixes (PR #87), `just
   real cities` and `just real languages`, scores published with the label
   error.
3. **Palier 2**: two facts per sentence, connectors, more question wordings
   (the second lever against template pairing), enrich `val`, regenerate,
   retrain Evaluate then Read on 3 seeds, release 0.0.3 on Franck's go.
4. **Learned Plan and Confront**: after Evaluate generalises; gains are
   first on `test`/`val`.
5. **Human correction beyond chains**: needs a hint that steers a search
   without supporting an answer.

## Later ideas (not scheduled)

- **Language packs**: languages downloaded on demand, English pre-installed
  beside the system language, consent once. Needs a shared core plus light
  per-language parts; test with French, core frozen, against a template
  translator.
- **Neutral form**: per-language readers into (subject, relation, value,
  certainty), the core reasoning on it beside the text.
- **Plugins**: data packs loaded like a library, tools run confined like
  programs; one package manager (manifest with the compatible core version,
  signature, consent).
- **Clarification**: an ambiguous request (two Charlestons) makes the core
  ask which one, through Frame's `ill_posed_question`.
- **Notebook memory**, **looped core**, **demo app** (the loop shown live on
  one subject), **footprint** per release (kWh, RAM, disk, parameters).
- Real text: keep the infobox at collection (birth years and release years
  live there), and a lever for questions sharing no word with the text.

## Open issues

| Issue | About | Belongs to |
|---|---|---|
| #74 | No memory cap on macOS; log peak RAM per run | training, footprint |
| #21 | CI: double runs, `check.sql`, uv cache | CI |
| #14 | CI: the training chain end to end on a tiny dataset | CI |
| #17 | worldgen: document the invariants behind `index`/`expect` | worldgen |
| #18 | search `k` and Confront's origin map | fixed by #75, to close |

Also pending: `[profile.test]` opt-level for worldgen (a manifest change),
dependabot (monthly, grouped), three worldgen files over ~200 lines
(`pronouns.rs`, `grammar.rs`, `traps.rs`).

## Moving to another machine

Git holds the code, the CHANGELOG and the step snapshots (`scores/`), but
only for branches pushed to `origin`: push every branch in progress first
(`git branch -vv` shows the ones with no remote), or copy the whole repo.
Everything else is rebuilt or copied:

| Not in git | Copy or rebuild |
|---|---|
| `data/ref-0.0.2/` (the 0.0.2 worlds, 214 MB, own `SHA256SUMS`) | **copy**; or `git checkout v0.0.2 && just gen` |
| `data/*.jsonl` (worlds of the current generator) | `just gen` (same seed and generator, same bytes); worldgen changed since 0.0.2 (pronouns), so these are not the worlds the current models learned on |
| `data/evaluate`, `data/read` (~540 MB, built from the 0.0.2 worlds) | **copy**, since the current cores and the overlap runs train on them; rebuilding needs those worlds loaded first (`just load` on them), then `just evaluate-data` and `just read-data`, which read the database |
| `data/real/` with its `.cache` (~57 MB) | **copy** the whole folder, then `shasum -a 256 -c SHA256SUMS` inside it (the checksums are not in git); collecting again fetches today's revisions, a different corpus |
| Postgres volume | `just up && just reset-db && just load` |
| `models/` (~40 MB) | **copy** to compare old and new on the same footing; retraining takes hours |
| `predictions/` | `just baseline`, `just core`, `just real` |
| `.env` | from `.env.example`; keep `POSTGRES_PORT=5442` |

Then, on the new machine:

- Install Docker, rustup (`rust-toolchain.toml` picks the version), uv and
  just; run `uv sync`.
- `just up && just test` before anything else.
- **Remeasure what depends on the hardware**: the thread bench (on the M3,
  training stops gaining past 6 threads: 0.96 s/step on 1, 0.445 on 6), the
  time per epoch (Evaluate full data: ~35 min on the M3 at 6 threads), and
  seed variance, which the M3 runs were meant to settle first.
- Session memory does not travel: what a fresh session must know is in this
  file, `CLAUDE.md`, the CHANGELOG and `docs/`.
