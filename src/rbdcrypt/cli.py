from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated
from zipfile import ZIP_DEFLATED, ZipFile

import typer

from .config import resolve_env_file
from .runtime.app import build_runtime
from .runtime.worker import RuntimeWorker


app = typer.Typer(add_completion=False, help="rbdcrypt futures signal/trading engine (paper-first)")


def _json_print(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _create_state_backup_zip(runtime, *, out_dir: Path, label: str | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label.strip()}" if label and label.strip() else ""
    backup_path = out_dir / f"rbdcrypt_state_backup_{ts}{suffix}.zip"

    env_file = resolve_env_file()
    db_path = runtime.settings.storage.db_path
    log_path = runtime.settings.logging.log_path
    meta = {
        "created_at_utc": datetime.now(tz=UTC).isoformat(),
        "db_path": str(db_path),
        "env_file": str(env_file) if env_file else None,
        "log_path": str(log_path),
        "label": label,
    }

    with ZipFile(backup_path, mode="w", compression=ZIP_DEFLATED) as zf:
        if db_path.is_file():
            zf.write(db_path, arcname="rbdcrypt.sqlite3")
        if env_file and env_file.is_file():
            zf.write(env_file, arcname=".env")
        if log_path.is_file():
            zf.write(log_path, arcname=f"logs/{log_path.name}")
        zf.writestr("meta.json", json.dumps(meta, indent=2, ensure_ascii=False))
    return backup_path


def _reset_runtime_state(runtime, *, include_ohlcv: bool) -> dict[str, object]:
    tables = [
        "signal_decisions",
        "signals",
        "market_context",
        "positions_active",
        "trades_closed",
        "errors",
        "heartbeats",
        "runtime_state",
    ]
    if include_ohlcv:
        tables.append("ohlcv_futures")

    now = datetime.now(tz=UTC)
    now_iso = now.isoformat()
    hour_anchor = now.strftime("%Y-%m-%dT%H:00:00+00:00")
    portfolio = {
        "balance": float(runtime.settings.balance.starting_balance),
        "realized_pnl": 0.0,
        "day_anchor": now_iso,
    }
    default_state = {
        "portfolio": portfolio,
        "cooldowns": {},
        "trade_missed_counters": {
            "hour_anchor": hour_anchor,
            "hourly_missed_signals": 0,
            "last_cycle_missed_signals": 0,
            "last_cycle_max_pos_blocked": 0,
            "updated_at": now_iso,
        },
        "notifications_state": {},
    }

    deleted: dict[str, int] = {}
    with runtime.db.transaction() as conn:
        for table in tables:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
            deleted[table] = int(row["c"] if row else 0)
        for table in tables:
            conn.execute(f"DELETE FROM {table}")
        for key, value in default_state.items():
            conn.execute(
                "INSERT INTO runtime_state (key, value_json, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value, ensure_ascii=True, separators=(",", ":")), now_iso),
            )

    return {
        "deleted_rows": deleted,
        "portfolio": portfolio,
        "include_ohlcv": include_ohlcv,
        "reset_at_utc": now_iso,
    }


def _runtime_recently_active(runtime) -> bool:
    threshold = timedelta(seconds=max(120, int(runtime.settings.runtime.scan_interval_sec) * 3))
    latest: datetime | None = None
    for component in ("scanner", "trader"):
        hb = runtime.repos.heartbeats.latest(component)
        if not hb:
            continue
        created = hb.get("created_at")
        if not isinstance(created, str) or not created:
            continue
        try:
            ts = datetime.fromisoformat(created)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        if latest is None or ts > latest:
            latest = ts
    if latest is None:
        return False
    return (datetime.now(tz=UTC) - latest) <= threshold


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


@app.command("backup-state")
def backup_state_cmd(
    out: Path = typer.Option(Path("bot_data/backups"), help="Backup output directory"),
    label: str | None = typer.Option(None, help="Optional label suffix for zip filename"),
) -> None:
    runtime = build_runtime()
    try:
        backup_path = _create_state_backup_zip(runtime, out_dir=out, label=label)
        _json_print({"backup_zip": str(backup_path)})
    finally:
        runtime.close()


@app.command("reset-state")
def reset_state_cmd(
    yes: bool = typer.Option(False, "--yes", help="Confirm destructive reset"),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create a zip backup before reset"),
    include_ohlcv: bool = typer.Option(
        True,
        "--include-ohlcv/--keep-ohlcv",
        help="Delete candle cache too (full clean reset)",
    ),
    out: Path = typer.Option(Path("bot_data/backups"), help="Backup output directory"),
    label: str | None = typer.Option("manual_reset", help="Optional backup label"),
    force_while_active: bool = typer.Option(
        False,
        "--force-while-active",
        help="Allow reset even if scanner/trader heartbeat looks active",
    ),
) -> None:
    if not yes:
        raise typer.BadParameter("Refusing to reset without --yes")
    runtime = build_runtime()
    try:
        if _runtime_recently_active(runtime) and not force_while_active:
            raise typer.BadParameter(
                "Runtime appears active. Stop service first (recommended), or use --force-while-active."
            )
        backup_path = _create_state_backup_zip(runtime, out_dir=out, label=label) if backup else None
        result = _reset_runtime_state(runtime, include_ohlcv=include_ohlcv)
        _json_print({"backup_zip": str(backup_path) if backup_path else None, **result})
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
