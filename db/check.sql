-- Smoke check: fails loudly if the schema did not initialise.
DO $$
DECLARE
  missing text;
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
    RAISE EXCEPTION 'extension vector is not installed';
  END IF;
  SELECT string_agg(t, ', ') INTO missing
  FROM unnest(ARRAY['worlds','entities','facts','sources','questions']) AS t
  WHERE to_regclass('public.' || t) IS NULL;
  IF missing IS NOT NULL THEN
    RAISE EXCEPTION 'missing tables: %', missing;
  END IF;
  RAISE NOTICE 'schema OK';
END $$;
