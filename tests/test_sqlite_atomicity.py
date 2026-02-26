from __future__ import annotations

from datetime import UTC, datetime

from rbdcrypt.models.signal import SignalDirection, SignalEvent


def test_db_transaction_rolls_back_on_exception(temp_db) -> None:
    try:
        with temp_db.transaction() as conn:
            conn.execute(
                "INSERT INTO runtime_state (key, value_json, updated_at) VALUES (?, ?, ?)",
                ("k1", "{}", datetime.now(tz=UTC).isoformat()),
            )
            raise RuntimeError("boom")
    except RuntimeError:
        pass

    with temp_db.read_only() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM runtime_state WHERE key = 'k1'").fetchone()
        assert int(row["c"]) == 0


def test_sqlite_wal_mode_enabled(temp_db) -> None:
    with temp_db.read_only() as conn:
        row = conn.execute("PRAGMA journal_mode;").fetchone()
        assert str(row[0]).lower() == "wal"


def test_signal_persistence_roundtrip(repos) -> None:
    sig = SignalEvent(
        symbol="BTCUSDT",
        interval="5m",
        bar_time=datetime(2026, 2, 20, 12, 0, tzinfo=UTC),
        direction=SignalDirection.LONG,
        price=100000.0,
        power_score=77.7,
        metrics={"adx": 25.0},
        power_breakdown={"trend": 20.0},
        candidate_pass=True,
        auto_pass=True,
    )
    signal_id = repos.signals.insert_signal(sig)
    assert signal_id > 0
    assert repos.signals.last_scan_time() is not None
