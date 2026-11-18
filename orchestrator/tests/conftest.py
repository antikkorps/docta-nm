"""Shared fixtures. Database tests roll back, so the dev database is untouched."""

import os

import psycopg
import pytest


@pytest.fixture
def conn():
    url = os.environ.get("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set: run `just up`, then the tests through `just test`")
    # force_rollback: any transaction the code under test opens becomes a
    # savepoint inside this one, so nothing ever reaches the dev database.
    with psycopg.connect(url) as connection, connection.transaction(force_rollback=True):
        yield connection
