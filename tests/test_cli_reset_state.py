from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from rbdcrypt.cli import (
    _archive_existing_db_files,
    _recreate_clean_database,
    _reset_runtime_state,
    _reset_state_files,
    _runtime_recently_active,
)
from rbdcrypt.config import AppSettings
from rbdcrypt.storage.repositories import build_repositories


def _runtime_stub(temp_db):
    settings = AppSettings(_env_file=None)
    settings.storage.db_path = temp_db.path
    repos = build_repositories(temp_db)
    return SimpleNamespace(db=temp_db, settings=settings, repos=repos)


def test_reset_runtime_state_resets_core_tables_and_defaults(temp_db) -> None:
    runtime = _runtime_stub(temp_db)

    with temp_db.transaction() as conn:
        conn.execute(
            "INSERT INTO runtime_state (key, value_json, updated_at) VALUES (?, ?, ?)",
            ("old_key", "{}", "2026-02-27T00:00:00+00:00"),
        )
        conn.execute(
            """
            INSERT INTO ohlcv_futures (
                symbol, interval, open_time, open, high, low, close, volume, close_time, source, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "BTCUSDT",
                "5m",
                "2026-02-27T00:00:00+00:00",
                100.0,
                101.0,
                99.0,
                100.5,
                10.0,
                "2026-02-27T00:04:59+00:00",
                "test",
                "2026-02-27T00:05:00+00:00",
            ),
        )
        conn.execute(
            "INSERT INTO heartbeats (component, status, meta_json, created_at) VALUES (?, ?, ?, ?)",
            ("scanner", "ok", "{}", "2026-02-27T00:00:00+00:00"),
        )

    report = _reset_runtime_state(runtime, include_ohlcv=False)

    assert report["include_ohlcv"] is False
    assert report["deleted_rows"]["runtime_state"] == 1
    assert report["deleted_rows"]["heartbeats"] == 1

    with temp_db.read_only() as conn:
        keys = {
            row["key"]: json.loads(row["value_json"])
            for row in conn.execute("SELECT key, value_json FROM runtime_state").fetchall()
        }
        ohlcv_count = int(conn.execute("SELECT COUNT(*) AS c FROM ohlcv_futures").fetchone()["c"])
        heartbeats_count = int(conn.execute("SELECT COUNT(*) AS c FROM heartbeats").fetchone()["c"])

    assert "old_key" not in keys
    assert set(keys.keys()) == {
        "portfolio",
        "cooldowns",
        "processed_signal_keys",
        "active_position_index",
        "recovery",
        "flap_counters",
        "trade_missed_counters",
        "active_profile",
        "risk_runtime_defaults",
        "notifications_state",
    }
    assert float(keys["portfolio"]["starting_balance"]) == 150.0
    assert float(keys["portfolio"]["balance"]) == 150.0
    assert float(keys["portfolio"]["realized_pnl"]) == 0.0
    assert ohlcv_count == 1
    assert heartbeats_count == 0


def test_reset_runtime_state_can_clear_ohlcv(temp_db) -> None:
    runtime = _runtime_stub(temp_db)
    with temp_db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO ohlcv_futures (
                symbol, interval, open_time, open, high, low, close, volume, close_time, source, fetched_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "ETHUSDT",
                "5m",
                "2026-02-27T00:00:00+00:00",
                2000.0,
                2010.0,
                1990.0,
                2005.0,
                20.0,
                "2026-02-27T00:04:59+00:00",
                "test",
                "2026-02-27T00:05:00+00:00",
            ),
        )

    report = _reset_runtime_state(runtime, include_ohlcv=True)

    assert report["deleted_rows"]["ohlcv_futures"] == 1
    with temp_db.read_only() as conn:
        ohlcv_count = int(conn.execute("SELECT COUNT(*) AS c FROM ohlcv_futures").fetchone()["c"])
    assert ohlcv_count == 0


def test_runtime_recently_active_detects_fresh_heartbeat(temp_db) -> None:
    runtime = _runtime_stub(temp_db)
    now = datetime.now(tz=UTC)
    with temp_db.transaction() as conn:
        conn.execute(
            "INSERT INTO heartbeats (component, status, meta_json, created_at) VALUES (?, ?, ?, ?)",
            ("scanner", "ok", "{}", now.isoformat()),
        )
    assert _runtime_recently_active(runtime) is True


def test_runtime_recently_active_ignores_stale_heartbeat(temp_db) -> None:
    runtime = _runtime_stub(temp_db)
    old = datetime.now(tz=UTC) - timedelta(minutes=10)
    with temp_db.transaction() as conn:
        conn.execute(
            "INSERT INTO heartbeats (component, status, meta_json, created_at) VALUES (?, ?, ?, ?)",
            ("trader", "ok", "{}", old.isoformat()),
        )
    assert _runtime_recently_active(runtime) is False


def test_archive_and_recreate_database_flow(tmp_path: Path) -> None:
    db_path = tmp_path / "rbdcrypt.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE sample(id INTEGER PRIMARY KEY, v TEXT)")
        conn.execute("INSERT INTO sample(v) VALUES ('x')")
        conn.commit()
    db_path.with_suffix(".sqlite3-wal").write_text("wal", encoding="utf-8")
    db_path.with_suffix(".sqlite3-shm").write_text("shm", encoding="utf-8")

    archived = _archive_existing_db_files(db_path=db_path, archive_dir=tmp_path / "archive")
    assert len(archived) >= 1

    _recreate_clean_database(db_path=db_path, wal=True, busy_timeout_ms=5000)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runtime_state'"
        ).fetchone()
    assert row is not None


def test_reset_state_files_resets_snapshot(tmp_path: Path) -> None:
    cfg = AppSettings(_env_file=None)
    cfg.storage.snapshot_path = tmp_path / "state" / "runtime_snapshot.json"
    cfg.storage.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.storage.snapshot_path.write_text('{"old": true}', encoding="utf-8")

    reset_files = _reset_state_files(cfg)

    assert reset_files == [str(cfg.storage.snapshot_path)]
    assert cfg.storage.snapshot_path.read_text(encoding="utf-8").strip() == "{}"
