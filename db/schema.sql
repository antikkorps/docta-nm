-- Applied once, on first start of an empty volume (docker-entrypoint-initdb.d).
-- Changing it requires `just reset-db` to take effect.

-- Installed now so a later phase can add embeddings without touching the infra.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE worlds (
  world_id   text PRIMARY KEY,
  seed       bigint NOT NULL,
  split      text NOT NULL CHECK (split IN ('train','val','test','ood'))
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
  obj        jsonb NOT NULL,          -- entity id or literal value
  PRIMARY KEY (world_id, fact_id)
);

CREATE TABLE sources (
  world_id    text REFERENCES worlds ON DELETE CASCADE,
  source_id   text,
  title       text NOT NULL,
  reliability real NOT NULL,
  body        text NOT NULL,
  fact_ids    text[] NOT NULL,        -- ground truth, never exposed to a model
  -- 'simple' config: names are invented, no language dictionary must alter them.
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
