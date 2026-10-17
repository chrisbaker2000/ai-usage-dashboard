"""Shared fixtures. All personas, ids, and tokens in this suite are synthetic."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from usagedash.db import Database

NOW = datetime(2026, 10, 3, 14, 0, tzinfo=UTC)


@pytest.fixture
def now() -> datetime:
    return NOW


@pytest.fixture
def database(tmp_path: Path) -> Database:
    database = Database(tmp_path / "data" / "usage.db")
    database.initialize()
    return database
