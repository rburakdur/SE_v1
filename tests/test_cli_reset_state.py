from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from rbdcrypt.cli import _reset_runtime_state, _runtime_recently_active
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
    assert set(keys.keys()) == {"portfolio", "cooldowns", "trade_missed_counters", "notifications_state"}
    assert float(keys["portfolio"]["balance"]) == 100.0
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
