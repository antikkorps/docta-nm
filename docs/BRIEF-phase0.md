# Docta-nm — Brief phase 0 pour Claude Code

## Contexte

Docta-nm est un projet de recherche : construire **de zéro** une IA frugale (peu de RAM, peu de disque) dont le noyau ne contient **aucun fait**. Le noyau porte une méthode : il sait qu'il ne sait pas, cherche dans des sources, confronte, pèse, et conclut en citant ses sources. Les faits vivent dans les sources, jamais dans les poids.

Spec de référence (vision, boucle, décisions) : https://claude.ai/artifact/M8Aj7PFp75ooF3dhXfaR4R

La phase 0 ne contient **aucun entraînement**. Elle construit tout ce qu'il faut autour du futur modèle pour pouvoir le mesurer dès le premier jour.

## Objectif de la phase 0

Avoir une chaîne complète et reproductible :

```
générateur de mondes fictifs → Postgres → orchestrateur (faux modèle) → évaluation → tableau de scores
```

**Définition de « terminé »** : sur une machine vierge avec Docker, Rust et Python,
`make up && make gen && make load && make baseline` affiche un tableau de scores pour les trois faux modèles décrits plus bas.

**Jalon visé par la phase suivante** (pour comprendre à quoi sert tout ça) : un noyau minuscule, entraîné de zéro, qui **s'abstient correctement** sur les questions sans réponse et fait mieux que les faux modèles.

## Hors périmètre

- Tout entraînement de modèle, tout GPU, tout modèle pré-entraîné ou appel à un LLM.
- La recherche vectorielle (pgvector est installé mais pas utilisé en phase 0).
- Les sources bruitées ou contradictoires (phase ultérieure, état Confronter).

## Décisions déjà prises

- Monorepo, hébergé sur Forgejo.
- Générateur de mondes en **Rust**.
- **Postgres dockerisé** (image `pgvector/pgvector:pg17`) pour les sources du bac à sable.
- Le JSONL reste le **format canonique** produit par le générateur ; Postgres en est une projection chargée depuis le JSONL.
- La boucle est portée par l'**orchestrateur** (machine à états explicite dans le code), pas par le modèle.
- Orchestrateur, chargeur et évaluation en **Python** (même langage que le futur entraînement PyTorch).
- Aucun LLM nulle part dans la génération des données, pour ne pas importer leurs biais.

## Arborescence

```
docta-nm/
├── docker-compose.yml        # postgres (pgvector/pgvector:pg17) + volume nommé
├── .env.example              # POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, DATABASE_URL
├── Makefile                  # up, down, gen, load, baseline, test
├── db/
│   └── schema.sql            # appliqué au démarrage du conteneur (docker-entrypoint-initdb.d)
├── worldgen/                 # crate Rust : docta-worldgen
├── orchestrator/             # paquet Python : docta_orchestrator
├── eval/                     # paquet Python : docta_eval
├── data/                     # JSONL générés (ignoré par git)
└── .forgejo/workflows/ci.yml # cargo test + pytest
```

## 1. Infra : Postgres dockerisé

- `docker-compose.yml` avec un service `db` : image `pgvector/pgvector:pg17`, port 5432 exposé en local uniquement (`127.0.0.1:5432:5432`), volume nommé, healthcheck `pg_isready`.
- `db/schema.sql` monté dans `/docker-entrypoint-initdb.d/`, active l'extension `vector` (même si elle n'est pas encore utilisée).
- Identifiants lus depuis `.env` (jamais commité) ; fournir `.env.example`.

### Schéma

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE worlds (
  world_id   text PRIMARY KEY,
  seed       bigint NOT NULL,
  split      text NOT NULL CHECK (split IN ('train','val','test'))
);

CREATE TABLE entities (
  world_id   text REFERENCES worlds ON DELETE CASCADE,
  entity_id  text,
  kind       text NOT NULL,
  name       text NOT NULL,
  PRIMARY KEY (world_id, entity_id)
);

CREATE TABLE facts (
  world_id   text REFERENCES worlds ON DELETE CASCADE,
  fact_id    text,
  subj       text NOT NULL,
  pred       text NOT NULL,
  obj        jsonb NOT NULL,          -- id d'entité ou valeur littérale
  PRIMARY KEY (world_id, fact_id)
);

CREATE TABLE sources (
  world_id    text REFERENCES worlds ON DELETE CASCADE,
  source_id   text,
  title       text NOT NULL,
  reliability real NOT NULL,
  body        text NOT NULL,
  fact_ids    text[] NOT NULL,        -- vérité terrain, jamais exposée au modèle
  tsv         tsvector GENERATED ALWAYS AS
                (to_tsvector('simple', title || ' ' || body)) STORED,
  PRIMARY KEY (world_id, source_id)
);
CREATE INDEX sources_tsv_idx ON sources USING gin (tsv);

CREATE TABLE questions (
  world_id          text REFERENCES worlds ON DELETE CASCADE,
  question_id       text,
  body              text NOT NULL,
  primitive         text NOT NULL,
  hops              int  NOT NULL,
  answerable        boolean NOT NULL,
  answer            text,
  missing           text,
  required_facts    text[] NOT NULL,
  required_sources  text[] NOT NULL,
  PRIMARY KEY (world_id, question_id)
);
```

La configuration FTS `simple` est voulue : les noms sont inventés, aucun dictionnaire de langue ne doit les transformer.

## 2. Générateur de mondes (Rust, `worldgen/`)

### CLI

```
docta-worldgen --seed 42 --worlds 1000 --out data/ \
  --split-ratio 0.8,0.1,0.1 --unanswerable-ratio 0.33 --max-hops 2
```

Produit `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`.

Crates suggérées : `clap`, `serde`/`serde_json`, `rand` + `rand_chacha` (RNG déterministe et portable).

### Format JSONL v0

Une ligne par enregistrement, trois types. Chaque ligne porte `"schema":"v0"`.

```json
{"schema":"v0","type":"world","world_id":"w_000123","seed":123,"split":"train",
 "entities":[{"id":"e1","kind":"ville","name":"Varnelle"},{"id":"e2","kind":"personne","name":"Orlan Teste"}],
 "facts":[{"id":"f1","subj":"e1","pred":"fondee_en","obj":1432},{"id":"f2","subj":"e2","pred":"habite","obj":"e1"}]}

{"schema":"v0","type":"source","world_id":"w_000123","source_id":"s1",
 "title":"Chronique de Varnelle","reliability":0.9,
 "text":"Varnelle fut fondée en 1432 au bord du fleuve.","facts":["f1"]}

{"schema":"v0","type":"question","world_id":"w_000123","question_id":"q1",
 "text":"En quelle année a été fondée la ville où habite Orlan Teste ?",
 "primitive":"enchainer","hops":2,"answerable":true,"answer":"1432",
 "required_facts":["f2","f1"],"required_sources":["s2","s1"]}

{"schema":"v0","type":"question","world_id":"w_000123","question_id":"q7",
 "text":"En quelle année a été fondée Brask ?",
 "primitive":"reperer_manque","hops":1,"answerable":false,"answer":null,
 "missing":"aucune source ne mentionne la date de fondation de Brask",
 "required_facts":[],"required_sources":[]}
```

### Règles

- **Déterminisme** : même graine et mêmes options ⇒ fichiers identiques octet pour octet. La graine de chaque monde dérive de la graine globale.
- **Noms inventés** par assemblage de syllabes, pour qu'aucun ne corresponde à un mot ou un lieu réel. Prévoir une petite liste d'exclusion.
- **Faits répartis** entre plusieurs sources (1 à 3 faits par source), pour qu'une question à 2 sauts demande souvent de croiser deux sources.
- **Textes des sources** produits par gabarits (templates) en français, avec quelques variantes de formulation par prédicat. Pas de LLM.
- **Questions** générées par gabarits, primitives v0 : `retrouver` (1 saut), `enchainer` (2 sauts), `comparer` (deux entités, un attribut numérique), `reperer_manque` (sans réponse).
- **Questions sans réponse** (environ un tiers, réglable) : elles portent sur un fait **absent de toutes les sources** du monde, sur une entité qui existe bien dans le monde. Le champ `missing` décrit ce qui manque.
- **Découpage train/val/test par monde**, jamais par question.

### Tests obligatoires

- Déterminisme (deux exécutions, même graine ⇒ même sortie).
- Pour toute question répondable : les `required_facts` sont bien couverts par les `required_sources`, et la réponse se déduit de ces faits.
- Pour toute question sans réponse : aucun fait de toutes les sources du monde ne permet d'y répondre.
- Aucun `world_id` n'apparaît dans deux splits.
- Chaque ligne JSONL est valide et porte `schema: v0`.

## 3. Chargeur (Python)

`python -m docta_orchestrator.load data/*.jsonl` : charge les JSONL dans Postgres (idempotent : un rechargement remplace le monde). Utiliser `psycopg` 3. `DATABASE_URL` lu depuis l'environnement.

## 4. Orchestrateur v0 (Python, `orchestrator/`)

### Machine à états

`Cadrer → Planifier → Consulter → Évaluer → (retour à Planifier si pas assez) → Confronter → Conclure`

Sorties de secours possibles depuis Cadrer, Évaluer et Confronter : `question_mal_posee`, `introuvable`, `contradiction_sans_arbitrage`.

- Chaque état est une fonction pure ou presque : il reçoit un état de travail, renvoie le suivant. L'orchestrateur impose les transitions ; le modèle ne fait que remplir chaque état.
- Limite du nombre de tours de la boucle Planifier ↔ Évaluer (paramètre, défaut 3).
- **Trace** : chaque question produit une trace JSON (états traversés, requêtes émises, extraits obtenus, décision finale), écrite dans `data/traces/`.

### Outil de recherche exposé au modèle

`search(world_id, query, k=5)` : recherche plein texte Postgres sur `sources` du monde concerné, classée par `ts_rank`. Ne renvoie que `source_id`, `title`, `body`, `reliability`. **Jamais** `fact_ids`.

### Interface « modèle »

Une interface Python (Protocol) avec une méthode par état. Le futur noyau entraîné l'implémentera ; en phase 0, trois faux modèles :

1. `RandomModel` : choix aléatoires (graine fixe).
2. `AlwaysAbstainModel` : s'abstient toujours.
3. `KeywordModel` : cherche avec les mots de la question, prend le meilleur extrait, en tire une réponse par règles simples ; s'abstient si la recherche ne renvoie rien de pertinent.

### Sortie

Un fichier de prédictions JSONL par modèle :

```json
{"question_id":"q1","world_id":"w_000123","abstained":false,"answer":"1432",
 "cited_sources":["s1","s2"],"confidence":0.6,"exit":null}
```

## 5. Évaluation (Python, `eval/`)

`python -m docta_eval data/test.jsonl predictions/*.jsonl` affiche un tableau par modèle :

- **Exactitude** sur les questions répondables (correspondance exacte normalisée).
- **Citations** : précision et rappel des `cited_sources` par rapport aux `required_sources`.
- **Abstention** : précision, rappel et F1 de l'abstention sur les questions sans réponse. C'est la métrique du premier jalon.
- Ventilation par `primitive` et par `hops`.

Sortie console lisible + export JSON des scores (pour le suivi d'expériences plus tard).

## Conventions

- Petits commits, un sujet par commit.
- `cargo fmt`, `cargo clippy`, `ruff` et `pytest` passent en CI (Forgejo Actions).
- README à la racine : le pourquoi en trois lignes, puis comment lancer `make up gen load baseline`.
- Ne jamais introduire de LLM, d'API externe ni de modèle pré-entraîné dans la chaîne.
- En cas d'ambiguïté sur un choix structurant, **demander à Franck** plutôt que trancher seul.

## Ordre de travail suggéré

1. Repo, `docker-compose.yml`, `schema.sql`, Makefile `up/down`.
2. Générateur : types, RNG, noms, mondes, faits.
3. Générateur : sources, questions, sans-réponse, splits + tests.
4. Chargeur JSONL → Postgres.
5. Évaluation (testée sur des prédictions écrites à la main).
6. Orchestrateur, outil `search`, trois faux modèles, traces.
7. `make baseline` de bout en bout, CI, README.
