from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer

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
        report = {
            "db_integrity": runtime.repos.maintenance.integrity_check(),
            "db_path": str(runtime.settings.storage.db_path),
            "disk_free_mb": disk["free_mb"],
            "disk_min_required_mb": runtime.settings.housekeeping.min_disk_free_mb,
            "disk_ok": disk["free_mb"] >= runtime.settings.housekeeping.min_disk_free_mb,
            "last_scan_time": last_scan.isoformat() if last_scan else None,
            "active_positions": runtime.repos.positions.count_active(),
            "ohlcv_rows_total": runtime.repos.candles.count(),
            "cooldown_symbols": sorted(cooldowns.keys()),
            "cooldown_count": len(cooldowns),
            "error_count_last_1h": runtime.repos.errors.count_since(now - timedelta(hours=1)),
            "trade_missed_counters": missed,
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
