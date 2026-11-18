set dotenv-load := true

# List recipes
default:
    @just --list

# Start Postgres and wait until it is healthy
up:
    docker compose up -d --wait

# Stop Postgres (data kept in the named volume)
down:
    docker compose down

# Verify the schema is initialised (extension + tables)
check-db:
    docker compose exec -T db psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f - < db/check.sql

# Drop the volume and re-run schema.sql (destroys all loaded data)
reset-db:
    docker compose down -v
    just up
