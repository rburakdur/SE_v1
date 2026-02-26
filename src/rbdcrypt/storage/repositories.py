from __future__ import annotations

import csv
import json
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

from ..models.error_event import ErrorEvent
from ..models.market_context import MarketContext
from ..models.ohlcv import OHLCVBar
from ..models.position import ActivePosition
from ..models.signal import SignalDecision, SignalEvent
from ..models.trade import ClosedTrade
from .db import Database


def _dt_to_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()


def _dt_from_str(value: str | None) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)


@contextmanager
def _tx(db: Database):
    with db.transaction() as conn:
        yield conn


class SignalRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_signal(self, signal: SignalEvent) -> int:
        with self.db.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO signals (
                    symbol, interval, bar_time, direction, price, power_score,
                    candidate_pass, auto_pass, blocked_reasons, metrics_json,
                    power_breakdown_json, meta_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.symbol,
                    signal.interval,
                    _dt_to_str(signal.bar_time),
                    signal.direction.value,
                    signal.price,
                    signal.power_score,
                    int(signal.candidate_pass),
                    int(signal.auto_pass),
                    _json_dumps(signal.blocked_reasons),
                    _json_dumps(signal.metrics),
                    _json_dumps(signal.power_breakdown),
                    _json_dumps(signal.meta),
                    _dt_to_str(signal.created_at),
                ),
            )
            return int(cur.lastrowid)

    def insert_decisions(self, signal_id: int | None, decisions: list[SignalDecision]) -> None:
        if not decisions:
            return
        with self.db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO signal_decisions (
                    signal_id, symbol, bar_time, stage, outcome,
                    blocked_reason, decision_payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        signal_id,
                        d.symbol,
                        _dt_to_str(d.bar_time),
                        d.stage,
                        d.outcome,
                        d.blocked_reason,
                        _json_dumps(d.decision_payload),
                        _dt_to_str(d.created_at),
                    )
                    for d in decisions
                ],
            )

    def last_scan_time(self) -> datetime | None:
        with self.db.read_only() as conn:
            row = conn.execute("SELECT MAX(created_at) AS created_at FROM signals").fetchone()
            if not row or row["created_at"] is None:
                return None
            return _dt_from_str(str(row["created_at"]))


class MarketContextRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, ctx: MarketContext) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO market_context (
                    symbol, interval, bar_time, trend_direction, trend_score,
                    chop_state, metrics_json, meta_json, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ctx.symbol,
                    ctx.interval,
                    _dt_to_str(ctx.bar_time),
                    ctx.trend_direction,
                    ctx.trend_score,
                    ctx.chop_state,
                    _json_dumps(ctx.metrics),
                    _json_dumps(ctx.meta),
                    _dt_to_str(ctx.fetched_at),
                ),
            )

    def latest(self, symbol: str = "BTCUSDT") -> MarketContext | None:
        with self.db.read_only() as conn:
            row = conn.execute(
                "SELECT * FROM market_context WHERE symbol = ? ORDER BY fetched_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if not row:
                return None
            return MarketContext(
                symbol=row["symbol"],
                interval=row["interval"],
                bar_time=_dt_from_str(row["bar_time"]) or datetime.now(tz=UTC),
                trend_direction=row["trend_direction"],
                trend_score=row["trend_score"],
                chop_state=row["chop_state"],
                metrics=json.loads(row["metrics_json"]),
                meta=json.loads(row["meta_json"]),
                fetched_at=_dt_from_str(row["fetched_at"]) or datetime.now(tz=UTC),
            )


class PositionRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_active(self, position: ActivePosition) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO positions_active (
                    position_id, symbol, side, qty, entry_price, initial_sl, initial_tp,
                    current_sl, current_tp, opened_at, recovered_at, entry_bar_time,
                    last_update_at, best_pnl_pct, current_pnl_pct, status, leverage,
                    notional, strategy_tag, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(position_id) DO UPDATE SET
                    symbol = excluded.symbol,
                    side = excluded.side,
                    qty = excluded.qty,
                    entry_price = excluded.entry_price,
                    initial_sl = excluded.initial_sl,
                    initial_tp = excluded.initial_tp,
                    current_sl = excluded.current_sl,
                    current_tp = excluded.current_tp,
                    opened_at = positions_active.opened_at,
                    recovered_at = COALESCE(excluded.recovered_at, positions_active.recovered_at),
                    entry_bar_time = excluded.entry_bar_time,
                    last_update_at = excluded.last_update_at,
                    best_pnl_pct = excluded.best_pnl_pct,
                    current_pnl_pct = excluded.current_pnl_pct,
                    status = excluded.status,
                    leverage = excluded.leverage,
                    notional = excluded.notional,
                    strategy_tag = excluded.strategy_tag,
                    meta_json = excluded.meta_json
                """,
                self._to_row(position),
            )

    def _to_row(self, p: ActivePosition) -> tuple[Any, ...]:
        return (
            p.position_id,
            p.symbol,
            p.side.value,
            p.qty,
            p.entry_price,
            p.initial_sl,
            p.initial_tp,
            p.current_sl,
            p.current_tp,
            _dt_to_str(p.opened_at),
            _dt_to_str(p.recovered_at) if p.recovered_at else None,
            _dt_to_str(p.entry_bar_time),
            _dt_to_str(p.last_update_at),
            p.best_pnl_pct,
            p.current_pnl_pct,
            p.status,
            p.leverage,
            p.notional,
            p.strategy_tag,
            _json_dumps(p.meta),
        )

    def list_active(self) -> list[ActivePosition]:
        with self.db.read_only() as conn:
            rows = conn.execute("SELECT * FROM positions_active ORDER BY opened_at ASC").fetchall()
        return [self._from_row(r) for r in rows]

    def get_active(self, position_id: str) -> ActivePosition | None:
        with self.db.read_only() as conn:
            row = conn.execute(
                "SELECT * FROM positions_active WHERE position_id = ?",
                (position_id,),
            ).fetchone()
            return self._from_row(row) if row else None

    def delete_active(self, position_id: str) -> None:
        with self.db.transaction() as conn:
            conn.execute("DELETE FROM positions_active WHERE position_id = ?", (position_id,))

    def count_active(self) -> int:
        with self.db.read_only() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM positions_active").fetchone()
            return int(row["c"]) if row else 0

    def _from_row(self, row: Any) -> ActivePosition:
        return ActivePosition(
            position_id=row["position_id"],
            symbol=row["symbol"],
            side=row["side"],
            qty=row["qty"],
            entry_price=row["entry_price"],
            initial_sl=row["initial_sl"],
            initial_tp=row["initial_tp"],
            current_sl=row["current_sl"],
            current_tp=row["current_tp"],
            opened_at=_dt_from_str(row["opened_at"]) or datetime.now(tz=UTC),
            recovered_at=_dt_from_str(row["recovered_at"]),
            entry_bar_time=_dt_from_str(row["entry_bar_time"]) or datetime.now(tz=UTC),
            last_update_at=_dt_from_str(row["last_update_at"]) or datetime.now(tz=UTC),
            best_pnl_pct=row["best_pnl_pct"],
            current_pnl_pct=row["current_pnl_pct"],
            status=row["status"],
            leverage=row["leverage"],
            notional=row["notional"],
            strategy_tag=row["strategy_tag"],
            meta=json.loads(row["meta_json"]),
        )


class TradeRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert_closed(self, trade: ClosedTrade) -> None:
        self._insert_closed_record(trade)

    def insert_closed_and_remove_active(self, trade: ClosedTrade, position_id: str) -> None:
        with self.db.transaction() as conn:
            self._insert_closed_record(trade, conn=conn)
            conn.execute("DELETE FROM positions_active WHERE position_id = ?", (position_id,))

    def _insert_closed_record(self, trade: ClosedTrade, conn=None) -> None:
        close_conn = False
        if conn is None:
            conn = self.db.connect()
            close_conn = True
        try:
            if close_conn:
                conn.execute("BEGIN;")
            conn.execute(
                """
                INSERT OR REPLACE INTO trades_closed (
                    trade_id, position_id, symbol, side, qty, entry_price, exit_price,
                    initial_sl, initial_tp, current_sl, current_tp, opened_at, closed_at,
                    entry_bar_time, exit_reason, pnl_pct, pnl_quote, rr_initial, fee_paid, meta_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade.trade_id,
                    trade.position_id,
                    trade.symbol,
                    trade.side.value,
                    trade.qty,
                    trade.entry_price,
                    trade.exit_price,
                    trade.initial_sl,
                    trade.initial_tp,
                    trade.current_sl,
                    trade.current_tp,
                    _dt_to_str(trade.opened_at),
                    _dt_to_str(trade.closed_at),
                    _dt_to_str(trade.entry_bar_time),
                    trade.exit_reason,
                    trade.pnl_pct,
                    trade.pnl_quote,
                    trade.rr_initial,
                    trade.fee_paid,
                    _json_dumps(trade.meta),
                ),
            )
            if close_conn:
                conn.commit()
        except Exception:
            if close_conn:
                conn.rollback()
            raise
        finally:
            if close_conn:
                conn.close()

    def summary(self) -> dict[str, float | int]:
        with self.db.read_only() as conn:
            row = conn.execute(
                """
                SELECT
                    COUNT(*) AS total_trades,
                    SUM(CASE WHEN pnl_quote > 0 THEN 1 ELSE 0 END) AS wins,
                    COALESCE(SUM(pnl_quote), 0) AS total_pnl_quote,
                    COALESCE(AVG(pnl_pct), 0) AS avg_pnl_pct
                FROM trades_closed
                """
            ).fetchone()
            if not row:
                return {"total_trades": 0, "wins": 0, "total_pnl_quote": 0.0, "avg_pnl_pct": 0.0}
            return {
                "total_trades": int(row["total_trades"] or 0),
                "wins": int(row["wins"] or 0),
                "total_pnl_quote": float(row["total_pnl_quote"] or 0.0),
                "avg_pnl_pct": float(row["avg_pnl_pct"] or 0.0),
            }


class ErrorRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, event: ErrorEvent) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO errors (source, error_type, message, traceback_single_line, context_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.source,
                    event.error_type,
                    event.message.replace("\n", "\\n"),
                    (event.traceback_single_line or "").replace("\n", "\\n") or None,
                    _json_dumps(event.context),
                    _dt_to_str(event.created_at),
                ),
            )

    def count_since(self, since: datetime) -> int:
        with self.db.read_only() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM errors WHERE created_at >= ?",
                (_dt_to_str(since),),
            ).fetchone()
            return int(row["c"] or 0)


class RuntimeStateRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def set_json(self, key: str, value: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO runtime_state (key, value_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value_json = excluded.value_json,
                    updated_at = excluded.updated_at
                """,
                (key, _json_dumps(value), _dt_to_str(datetime.now(tz=UTC))),
            )

    def get_json(self, key: str) -> dict[str, Any] | None:
        with self.db.read_only() as conn:
            row = conn.execute(
                "SELECT value_json FROM runtime_state WHERE key = ?",
                (key,),
            ).fetchone()
            if not row:
                return None
            return json.loads(row["value_json"])


class HeartbeatRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def insert(self, component: str, status: str, meta: dict[str, Any]) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                "INSERT INTO heartbeats (component, status, meta_json, created_at) VALUES (?, ?, ?, ?)",
                (component, status, _json_dumps(meta), _dt_to_str(datetime.now(tz=UTC))),
            )

    def latest(self, component: str) -> dict[str, Any] | None:
        with self.db.read_only() as conn:
            row = conn.execute(
                "SELECT * FROM heartbeats WHERE component = ? ORDER BY created_at DESC LIMIT 1",
                (component,),
            ).fetchone()
            if not row:
                return None
            return {
                "component": row["component"],
                "status": row["status"],
                "meta": json.loads(row["meta_json"]),
                "created_at": row["created_at"],
            }


class MaintenanceRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def prune(self, *, table: str, date_column: str, retention_days: int) -> int:
        cutoff = datetime.now(tz=UTC) - timedelta(days=retention_days)
        with self.db.transaction() as conn:
            cur = conn.execute(
                f"DELETE FROM {table} WHERE {date_column} < ?",
                (_dt_to_str(cutoff),),
            )
            return int(cur.rowcount or 0)

    def export_csv(self, *, table: str, out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db.read_only() as conn:
            cur = conn.execute(f"SELECT * FROM {table}")
            rows = cur.fetchall()
            fieldnames = [col[0] for col in cur.description or []]
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        return out_path

    def integrity_check(self) -> str:
        with self.db.read_only() as conn:
            row = conn.execute("PRAGMA integrity_check;").fetchone()
            return str(row[0]) if row else "unknown"


class CandleRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_many(self, bars: list[OHLCVBar]) -> int:
        if not bars:
            return 0
        with self.db.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO ohlcv_futures (
                    symbol, interval, open_time, open, high, low, close, volume,
                    close_time, source, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    close_time = COALESCE(excluded.close_time, ohlcv_futures.close_time),
                    source = excluded.source,
                    fetched_at = excluded.fetched_at
                """,
                [
                    (
                        b.symbol,
                        b.interval,
                        _dt_to_str(b.open_time),
                        b.open,
                        b.high,
                        b.low,
                        b.close,
                        b.volume,
                        _dt_to_str(b.close_time) if b.close_time else None,
                        b.source,
                        _dt_to_str(b.fetched_at),
                    )
                    for b in bars
                ],
            )
            return len(bars)

    def list_bars(
        self,
        *,
        symbol: str,
        interval: str = "5m",
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int | None = None,
        ascending: bool = True,
    ) -> list[OHLCVBar]:
        clauses = ["symbol = ?", "interval = ?"]
        params: list[Any] = [symbol, interval]
        if start is not None:
            clauses.append("open_time >= ?")
            params.append(_dt_to_str(start))
        if end is not None:
            clauses.append("open_time <= ?")
            params.append(_dt_to_str(end))
        order = "ASC" if ascending else "DESC"
        sql = f"SELECT * FROM ohlcv_futures WHERE {' AND '.join(clauses)} ORDER BY open_time {order}"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        with self.db.read_only() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [self._from_row(r) for r in rows]

    def latest_open_time(self, *, symbol: str, interval: str = "5m") -> datetime | None:
        with self.db.read_only() as conn:
            row = conn.execute(
                "SELECT MAX(open_time) AS open_time FROM ohlcv_futures WHERE symbol = ? AND interval = ?",
                (symbol, interval),
            ).fetchone()
            if not row or row["open_time"] is None:
                return None
            return _dt_from_str(str(row["open_time"]))

    def count(self, *, symbol: str | None = None, interval: str | None = None) -> int:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if interval is not None:
            clauses.append("interval = ?")
            params.append(interval)
        sql = "SELECT COUNT(*) AS c FROM ohlcv_futures"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        with self.db.read_only() as conn:
            row = conn.execute(sql, tuple(params)).fetchone()
            return int(row["c"] or 0) if row else 0

    def _from_row(self, row: Any) -> OHLCVBar:
        return OHLCVBar(
            symbol=row["symbol"],
            interval=row["interval"],
            open_time=_dt_from_str(row["open_time"]) or datetime.now(tz=UTC),
            open=row["open"],
            high=row["high"],
            low=row["low"],
            close=row["close"],
            volume=row["volume"],
            close_time=_dt_from_str(row["close_time"]),
            source=row["source"],
            fetched_at=_dt_from_str(row["fetched_at"]) or datetime.now(tz=UTC),
        )


@dataclass(slots=True)
class Repositories:
    signals: SignalRepository
    market_context: MarketContextRepository
    positions: PositionRepository
    trades: TradeRepository
    errors: ErrorRepository
    runtime_state: RuntimeStateRepository
    heartbeats: HeartbeatRepository
    maintenance: MaintenanceRepository
    candles: CandleRepository


def build_repositories(db: Database) -> Repositories:
    return Repositories(
        signals=SignalRepository(db),
        market_context=MarketContextRepository(db),
        positions=PositionRepository(db),
        trades=TradeRepository(db),
        errors=ErrorRepository(db),
        runtime_state=RuntimeStateRepository(db),
        heartbeats=HeartbeatRepository(db),
        maintenance=MaintenanceRepository(db),
        candles=CandleRepository(db),
    )
