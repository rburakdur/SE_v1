from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

from .config import resolve_env_file
from .runtime.app import build_runtime
from .runtime.worker import RuntimeWorker


app = typer.Typer(add_completion=False, help="rbdcrypt futures signal/trading engine (paper-first)")


def _json_print(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


@app.command()
def run(
    one_shot: bool = typer.Option(False, help="Run a single scan/trade cycle and exit"),
    iterations: int | None = typer.Option(None, help="Run fixed number of cycles (testing)"),
) -> None:
    runtime = build_runtime()
    try:
        worker = RuntimeWorker(runtime)
        worker.run(one_shot=one_shot, max_iterations=iterations)
    finally:
        runtime.close()


def _parse_dt(value: str | None) -> datetime | None:
    if value is None:
        return None
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def _portfolio_snapshot(runtime) -> dict[str, object]:
    starting = float(runtime.settings.balance.starting_balance)
    portfolio = runtime.repos.runtime_state.get_json("portfolio") or {}
    balance = float(portfolio.get("balance", starting))
    realized_pnl = float(portfolio.get("realized_pnl", balance - starting))
    pnl_pct_cumulative = ((balance - starting) / starting * 100.0) if starting > 0 else 0.0
    day_anchor = portfolio.get("day_anchor")
    summary = runtime.repos.trades.summary()
    with runtime.db.read_only() as conn:
        row = conn.execute(
            "SELECT updated_at FROM runtime_state WHERE key = 'portfolio'",
        ).fetchone()
    updated_at = row["updated_at"] if row else None
    consistency_warning = None
    if int(summary["total_trades"]) == 0 and abs(balance - starting) > 1e-9:
        consistency_warning = "portfolio balance differs from starting_balance but trades_closed is empty"
    return {
        "starting_balance": starting,
        "balance": balance,
        "realized_pnl": realized_pnl,
        "pnl_pct_cumulative": pnl_pct_cumulative,
        "day_anchor": day_anchor,
        "portfolio_updated_at": updated_at,
        "consistency_warning": consistency_warning,
    }


def _decision_reason_summary(runtime, *, limit: int = 200) -> list[dict[str, object]]:
    sql = """
        WITH recent AS (
            SELECT stage, outcome, blocked_reason
            FROM signal_decisions
            ORDER BY id DESC
            LIMIT ?
        )
        SELECT
            stage,
            COALESCE(blocked_reason, '-') AS blocked_reason,
            COUNT(*) AS count
        FROM recent
        WHERE outcome = 'blocked'
        GROUP BY stage, COALESCE(blocked_reason, '-')
        ORDER BY count DESC, stage ASC, blocked_reason ASC
    """
    with runtime.db.read_only() as conn:
        rows = conn.execute(sql, (int(limit),)).fetchall()
    return [
        {
            "stage": r["stage"],
            "blocked_reason": r["blocked_reason"],
            "count": int(r["count"]),
        }
        for r in rows
    ]


def _recent_blocked_decisions(runtime, *, limit: int = 20) -> list[dict[str, object]]:
    sql = """
        SELECT id, created_at, symbol, stage, blocked_reason, decision_payload_json
        FROM signal_decisions
        WHERE outcome = 'blocked'
        ORDER BY id DESC
        LIMIT ?
    """
    with runtime.db.read_only() as conn:
        rows = conn.execute(sql, (int(limit),)).fetchall()
    out: list[dict[str, object]] = []
    for r in rows:
        payload = json.loads(r["decision_payload_json"]) if r["decision_payload_json"] else {}
        payload_hint = {
            "rejection_stage": payload.get("rejection_stage"),
            "candidate_score": payload.get("candidate_score"),
            "auto_score": payload.get("auto_score"),
            "effective_auto_power": payload.get("effective_auto_power"),
            "flags": payload.get("flags"),
            "thresholds": payload.get("thresholds"),
        }
        out.append(
            {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "symbol": r["symbol"],
                "stage": r["stage"],
                "blocked_reason": r["blocked_reason"],
                "payload_hint": payload_hint,
            }
        )
    return out


def _recent_signals(runtime, *, limit: int = 20) -> list[dict[str, object]]:
    sql = """
        SELECT id, created_at, symbol, direction, price, power_score, candidate_pass, auto_pass, blocked_reasons
        FROM signals
        ORDER BY id DESC
        LIMIT ?
    """
    with runtime.db.read_only() as conn:
        rows = conn.execute(sql, (int(limit),)).fetchall()
    out: list[dict[str, object]] = []
    for r in rows:
        blocked = json.loads(r["blocked_reasons"]) if r["blocked_reasons"] else []
        out.append(
            {
                "id": int(r["id"]),
                "created_at": r["created_at"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "price": float(r["price"]),
                "power_score": float(r["power_score"]),
                "candidate_pass": bool(r["candidate_pass"]),
                "auto_pass": bool(r["auto_pass"]),
                "blocked_reasons": blocked,
            }
        )
    return out


@app.command()
def backfill(
    symbols: str = typer.Option("top:10", help="Comma list (BTCUSDT,ETHUSDT) or top[:N]"),
    days: float | None = typer.Option(3.0, help="How many days of candles to fetch (ignored if --bars)"),
    bars: int | None = typer.Option(None, help="How many bars to fetch per symbol"),
    include_btc: bool = typer.Option(True, help="Always include BTC benchmark candles"),
    incremental: bool = typer.Option(True, help="Start from latest stored candle if available"),
    chunk_limit: int = typer.Option(1000, min=1, max=1500, help="Binance kline batch size"),
) -> None:
    runtime = build_runtime()
    try:
        resolved = runtime.backfill_service.resolve_symbols(symbols, include_btc=include_btc)
        summary = runtime.backfill_service.backfill(
            symbols=resolved,
            interval=runtime.settings.binance.interval,
            bars=bars,
            days=days,
            chunk_limit=chunk_limit,
            incremental=incremental,
        )
        _json_print(
            {
                "symbols": summary.symbols,
                "interval": summary.interval,
                "inserted_bars": summary.inserted_bars,
                "by_symbol": summary.by_symbol,
            }
        )
    finally:
        runtime.close()


@app.command()
def replay(
    symbol: str = typer.Argument(..., help="Symbol to replay from local backfill data"),
    days: float | None = typer.Option(None, help="Replay only last N days"),
    start: Annotated[str | None, typer.Option(help="Start datetime (ISO8601, UTC preferred)")] = None,
    end: Annotated[str | None, typer.Option(help="End datetime (ISO8601, UTC preferred)")] = None,
    warmup_bars: int = typer.Option(80, min=60, help="Warmup bars before evaluating signals"),
    max_trades: int | None = typer.Option(None, min=1, help="Stop after N closed trades"),
    persist_report: bool = typer.Option(True, help="Store replay summary into runtime_state"),
) -> None:
    runtime = build_runtime()
    try:
        end_dt = _parse_dt(end)
        start_dt = _parse_dt(start)
        if days is not None and start_dt is None:
            ref_end = end_dt or datetime.now(tz=UTC)
            start_dt = ref_end - timedelta(days=days)
        report = runtime.replay_service.replay_symbol(
            symbol=symbol.upper(),
            interval=runtime.settings.binance.interval,
            start=start_dt,
            end=end_dt,
            warmup_bars=warmup_bars,
            max_trades=max_trades,
            persist_report=persist_report,
        )
        _json_print(
            {
                "symbol": report.symbol,
                "interval": report.interval,
                "start": report.start,
                "end": report.end,
                "bars_used": report.bars_used,
                "bars_skipped_unaligned": report.bars_skipped_unaligned,
                "candidate_signals": report.candidate_signals,
                "auto_signals": report.auto_signals,
                "opened": report.opened,
                "closed": report.closed,
                "missed": report.missed,
                "wins": report.wins,
                "losses": report.losses,
                "total_pnl_quote": round(report.total_pnl_quote, 6),
                "cooldown_blocks": report.cooldown_blocks,
                "trades_preview": [asdict(t) if is_dataclass(t) else t for t in report.trades[:10]],
            }
        )
    finally:
        runtime.close()


@app.command()
def analyze() -> None:
    runtime = build_runtime()
    try:
        _json_print(runtime.metrics_service.analyze_summary())
    finally:
        runtime.close()


@app.command()
def doctor() -> None:
    runtime = build_runtime()
    try:
        now = datetime.now(tz=UTC)
        disk = runtime.housekeeping_service.disk_status()
        last_scan = runtime.repos.signals.last_scan_time()
        api_ok = runtime.binance_client.ping() if runtime.settings.runtime.enable_api_connectivity_check else None
        cooldowns = runtime.repos.runtime_state.get_json("cooldowns") or {}
        missed = runtime.repos.runtime_state.get_json("trade_missed_counters") or {}
        portfolio = _portfolio_snapshot(runtime)
        trade_summary = runtime.repos.trades.summary()
        env_file = resolve_env_file()
        report = {
            "db_integrity": runtime.repos.maintenance.integrity_check(),
            "db_path": str(runtime.settings.storage.db_path),
            "config_env_file": str(env_file) if env_file else None,
            "config_env_file_exists": bool(env_file and env_file.is_file()),
            "disk_free_mb": disk["free_mb"],
            "disk_min_required_mb": runtime.settings.housekeeping.min_disk_free_mb,
            "disk_ok": disk["free_mb"] >= runtime.settings.housekeeping.min_disk_free_mb,
            "last_scan_time": last_scan.isoformat() if last_scan else None,
            "active_positions": runtime.repos.positions.count_active(),
            "ohlcv_rows_total": runtime.repos.candles.count(),
            "portfolio": portfolio,
            "trades_closed_summary": {
                "total_trades": int(trade_summary["total_trades"]),
                "wins": int(trade_summary["wins"]),
                "losses": int(trade_summary["total_trades"]) - int(trade_summary["wins"]),
                "total_pnl_quote": float(trade_summary["total_pnl_quote"]),
                "avg_pnl_pct": float(trade_summary["avg_pnl_pct"]),
            },
            "cooldown_symbols": sorted(cooldowns.keys()),
            "cooldown_count": len(cooldowns),
            "error_count_last_1h": runtime.repos.errors.count_since(now - timedelta(hours=1)),
            "trade_missed_counters": missed,
            "blocked_reasons_last_200": _decision_reason_summary(runtime, limit=200),
            "recent_blocked_decisions": _recent_blocked_decisions(runtime, limit=15),
            "api_connectivity": api_ok,
            "notifications": {
                "enabled": runtime.settings.notifications.enabled,
                "topic": runtime.settings.notifications.topic,
                "url": runtime.settings.notifications.ntfy_url,
            },
            "scanner_heartbeat": runtime.repos.heartbeats.latest("scanner"),
            "trader_heartbeat": runtime.repos.heartbeats.latest("trader"),
        }
        _json_print(report)
    finally:
        runtime.close()


@app.command("why-no-trade")
def why_no_trade(
    limit: int = typer.Option(100, min=20, max=1000, help="How many recent rows to inspect"),
) -> None:
    runtime = build_runtime()
    try:
        trade_summary = runtime.repos.trades.summary()
        report = {
            "portfolio": _portfolio_snapshot(runtime),
            "active_positions": runtime.repos.positions.count_active(),
            "trades_closed_total": int(trade_summary["total_trades"]),
            "blocked_reasons_last_n": _decision_reason_summary(runtime, limit=limit),
            "recent_blocked_decisions": _recent_blocked_decisions(runtime, limit=min(limit, 30)),
            "recent_signals": _recent_signals(runtime, limit=min(limit, 30)),
        }
        _json_print(report)
    finally:
        runtime.close()


@app.command("rotate")
def rotate_cmd() -> None:
    runtime = build_runtime()
    try:
        _json_print({"deleted": runtime.housekeeping_service.prune()})
    finally:
        runtime.close()


@app.command("prune")
def prune_cmd() -> None:
    rotate_cmd()


@app.command("export-csv")
def export_csv(
    table: str = typer.Argument(..., help="Table name to export"),
    out: Path = typer.Option(Path("bot_data/exports"), help="Output directory"),
) -> None:
    allowed = {
        "signals",
        "signal_decisions",
        "market_context",
        "positions_active",
        "trades_closed",
        "errors",
        "runtime_state",
        "heartbeats",
        "ohlcv_futures",
    }
    if table not in allowed:
        raise typer.BadParameter(f"Unsupported table '{table}'. Allowed: {sorted(allowed)}")
    runtime = build_runtime()
    try:
        out_file = runtime.repos.maintenance.export_csv(table=table, out_path=out / f"{table}.csv")
        typer.echo(str(out_file))
    finally:
        runtime.close()


if __name__ == "__main__":
    app()
