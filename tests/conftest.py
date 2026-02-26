from __future__ import annotations

from pathlib import Path

import pytest

from rbdcrypt.storage.db import Database
from rbdcrypt.storage.migrations import apply_migrations
from rbdcrypt.storage.repositories import build_repositories


@pytest.fixture()
def temp_db(tmp_path: Path) -> Database:
    db = Database(tmp_path / "test.sqlite3", wal=True)
    apply_migrations(db)
    return db


@pytest.fixture()
def repos(temp_db: Database):
    return build_repositories(temp_db)
