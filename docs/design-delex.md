# Real structures, invented facts (delexicalised training text)

Decided with Franck on 2026-11-28. The first real-text run (0.0.2 cores) read
nothing: 39–46% of the tokens were unknown and every input was longer than
the core reads. The core has to learn real English. Training on real text as
it is would put real facts in its weights, the one thing the project rules
out, and would spend the probe, our only honest measure of transfer.

So the core learns from real sentences whose facts were replaced: the
structure, vocabulary, length and noise of Wikipedia prose, around names and
values that exist nowhere but in its sources.

## Corpus

- Three domains outside the probe's (cities, programming languages), close to
  the worlds' predicates and with no living people as subjects:
  **universities** (founded, city, students), **mountains** (elevation,
  range, first ascent), **bridges** (opened, length, what it crosses).
- About 1,000 articles per domain, collected by the probe's collector,
  pinned to their revision, with each article's internal links (the API's
  `prop=links`) kept beside its text.
- Wikipedia is CC BY-SA 4.0. The corpus stays local and gitignored, like the
  probe, and is used for training only.

## What is replaced, without a model

No pretrained model may find the names (no NER), so spans come from what the
sources give:

- **Entities**: the article's subject, its Wikidata aliases, and the titles
  of its internal links (and their labels without the parenthesised
  disambiguation), matched in the plain text, longest first.
- **Years and other numbers**: regular expressions.
- **Common nouns stay** ("Puritan settlers"): they carry the language, not
  the facts the questions ask about.

## What replaces it

- **Names**: invented by syllable assembly, from the worldgen crate (a new
  subcommand prints names for a seed), so one generator writes every name in
  the project. The same original gets the same name across the article; a
  multi-word original gets a name of as many words where it can.
- **Years**: one offset per article drawn from its seed, so order and gaps
  hold ("founded in 1749, incorporated in 1779" keeps its 30 years) while no
  date is the real one.
- **Other numbers** (elevations, lengths, students): one factor per article
  and kind, so magnitudes stay plausible.
- Determinism: same corpus, same seed, same bytes (ChaCha8, ordered maps).

## Labels

The probe's machinery (`realtext.questions`, `stated`, `redact`) runs on the
delexicalised articles with the same replacements applied to the Wikidata
facts: questions in the worlds' wordings, answerable, naturally missing or
redacted. The values were put there by us, so the labels are exact where the
probe's were checked by hand.

## Leak control

The main risk is a mention the rules missed ("Boston" with no link), which
would let a real fact through.

- **Automatic**: no label or alias of a replaced entity, and no original
  value of a replaced fact, may survive in the output; the rate is reported
  and a threshold blocks the build.
- **By hand**: a page of 50 passages, original beside delexicalised, that
  Franck checks, as for the probe's labels.

## Splits and loading

- Split by article, never by sentence: `delex-train`, `delex-val`,
  `delex-test`, one world per domain and split, loaded by the real-text
  loader. These are three new values in the `split` check of
  `db/schema.sql` (agreed with Franck); `--prune` leaves them alone, as it
  leaves the probe.
- `delex-val` is the real-language dev set: settings that touch real text
  (BM25 variants, passage length) are chosen on it, never on the probe.

## Training and measures

- Evaluate and Read are retrained on the invented worlds (the method, traps,
  sealed `ood`) plus `delex-train` (the language), the vocabulary relearned
  on both, the overlap flag on, 3 seeds.
- Reported: test, `val` and `ood` of the worlds, `delex-val`, then the probe
  once.

## Steps (one PR each)

1. Collector for the three domains, links included.
2. Name subcommand in worldgen; the delexicaliser (spans, consistent
   replacements, offsets), test-first and deterministic.
3. Leak control: the automatic check, then the hand-checked page.
4. Questions and labels on the delexicalised corpus.
5. Schema values, loading, Evaluate and Read data; training on 3 seeds.
6. Measures as above.
