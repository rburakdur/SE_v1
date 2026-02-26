from __future__ import annotations

from .db import Database


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    direction TEXT NOT NULL,
    price REAL NOT NULL,
    power_score REAL NOT NULL,
    candidate_pass INTEGER NOT NULL,
    auto_pass INTEGER NOT NULL,
    blocked_reasons TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    power_breakdown_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_bar_time ON signals(symbol, bar_time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_created_at ON signals(created_at DESC);

CREATE TABLE IF NOT EXISTS signal_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER,
    symbol TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    stage TEXT NOT NULL,
    outcome TEXT NOT NULL,
    blocked_reason TEXT,
    decision_payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(signal_id) REFERENCES signals(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_signal_decisions_symbol_bar_time ON signal_decisions(symbol, bar_time DESC);

CREATE TABLE IF NOT EXISTS market_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    bar_time TEXT NOT NULL,
    trend_direction TEXT NOT NULL,
    trend_score REAL NOT NULL,
    chop_state TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_market_context_symbol_fetched_at ON market_context(symbol, fetched_at DESC);

CREATE TABLE IF NOT EXISTS positions_active (
    position_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    initial_sl REAL NOT NULL,
    initial_tp REAL NOT NULL,
    current_sl REAL NOT NULL,
    current_tp REAL NOT NULL,
    opened_at TEXT NOT NULL,
    recovered_at TEXT,
    entry_bar_time TEXT NOT NULL,
    last_update_at TEXT NOT NULL,
    best_pnl_pct REAL NOT NULL,
    current_pnl_pct REAL NOT NULL,
    status TEXT NOT NULL,
    leverage REAL NOT NULL,
    notional REAL NOT NULL,
    strategy_tag TEXT NOT NULL,
    meta_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_positions_active_symbol ON positions_active(symbol);

CREATE TABLE IF NOT EXISTS trades_closed (
    trade_id TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    initial_sl REAL NOT NULL,
    initial_tp REAL NOT NULL,
    current_sl REAL NOT NULL,
    current_tp REAL NOT NULL,
    opened_at TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    entry_bar_time TEXT NOT NULL,
    exit_reason TEXT NOT NULL,
    pnl_pct REAL NOT NULL,
    pnl_quote REAL NOT NULL,
    rr_initial REAL NOT NULL,
    fee_paid REAL NOT NULL,
    meta_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_trades_closed_closed_at ON trades_closed(closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_trades_closed_symbol ON trades_closed(symbol, closed_at DESC);

CREATE TABLE IF NOT EXISTS errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    error_type TEXT NOT NULL,
    message TEXT NOT NULL,
    traceback_single_line TEXT,
    context_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_errors_created_at ON errors(created_at DESC);

CREATE TABLE IF NOT EXISTS runtime_state (
    key TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_heartbeats_component_created_at ON heartbeats(component, created_at DESC);

CREATE TABLE IF NOT EXISTS ohlcv_futures (
    symbol TEXT NOT NULL,
    interval TEXT NOT NULL,
    open_time TEXT NOT NULL,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume REAL NOT NULL,
    close_time TEXT,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (symbol, interval, open_time)
);

CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_interval_open_time ON ohlcv_futures(symbol, interval, open_time DESC);
"""


def apply_migrations(db: Database) -> None:
    with db.transaction() as conn:
        conn.executescript(SCHEMA_SQL)
