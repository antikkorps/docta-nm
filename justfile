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

# Generate the fictional-world dataset into data/ (same options = same bytes)
gen seed="42" worlds="1000":
    cargo run --release -q -p docta-worldgen -- --seed {{seed}} --worlds {{worlds}} --out data \
        --split-ratio 0.8,0.1,0.1 --unanswerable-ratio 0.33 --max-hops 2

# Load data/*.jsonl into Postgres, mirroring it: worlds no longer in data/ are removed
load:
    uv run python -m docta_orchestrator.load --prune data/*.jsonl

# Every check run before pushing (needs `just up` for the database tests)
test:
    @test -n "${DATABASE_URL:-}" || { echo "DATABASE_URL missing from .env: the database tests would be skipped"; exit 1; }
    cargo fmt --check
    cargo clippy --all-targets -- -D warnings
    cargo test -q
    uv run ruff check
    uv run ruff format --check
    uv run pytest -q
