from __future__ import annotations

import traceback
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Callable

from ..config import AppSettings
from ..data.market_fetcher import MarketFetcher
from ..models.error_event import ErrorEvent
from ..models.market_context import MarketContext
from ..models.signal import SignalEvent
from ..models.symbol_state import SymbolBarState
from ..notifications.ntfy_client import NtfyClient
from ..notifications.notification_service import NotificationService
from ..storage.repositories import Repositories
from ..strategy.parity_signal_engine import ParitySignalEngine


@dataclass(slots=True)
class ScanRunResult:
    started_at: datetime
    finished_at: datetime
    btc_context: MarketContext | None
    signals: list[SignalEvent] = field(default_factory=list)
    prices_by_symbol: dict[str, float] = field(default_factory=dict)
    symbol_states: dict[str, SymbolBarState] = field(default_factory=dict)
    scanned_symbols: int = 0
    error_count: int = 0

    @property
    def auto_signals(self) -> list[SignalEvent]:
        return [s for s in self.signals if s.auto_pass]


class ScanService:
    def __init__(
        self,
        *,
        settings: AppSettings,
        fetcher: MarketFetcher,
        signal_engine: ParitySignalEngine,
        repos: Repositories,
        now_fn: Callable[[], datetime],
        logger,
        notifier: NtfyClient | None = None,
        notification_service: NotificationService | None = None,
    ) -> None:
        self.settings = settings
        self.fetcher = fetcher
        self.signal_engine = signal_engine
        self.repos = repos
        self.now_fn = now_fn
        self.logger = logger
        self.notifier = notifier
        self.notification_service = notification_service
        self._last_cycle_summary_notify_at: datetime | None = None

    def scan_once(self) -> ScanRunResult:
        started = self.now_fn()
        btc_ctx: MarketContext | None = None
        signals: list[SignalEvent] = []
        prices: dict[str, float] = {}
        symbol_states: dict[str, SymbolBarState] = {}
        error_count = 0
        error_samples: list[str] = []

        try:
            btc_candles = self.fetcher.fetch_btc_context_candles()
            btc_ctx, btc_symbol_state, _ = self.signal_engine.derive_btc_market_context(btc_candles)
            self.repos.market_context.insert(btc_ctx)
            symbol_states[btc_symbol_state.symbol] = btc_symbol_state
        except Exception as exc:
            error_count += 1
            error_samples.append(f"btc_context:{exc.__class__.__name__}")
            self._record_error("scan_service.btc_context", exc, {})

        try:
            symbols = self.fetcher.fetch_universe_symbols()
        except Exception as exc:
            error_count += 1
            error_samples.append(f"fetch_universe:{exc.__class__.__name__}")
            self._record_error("scan_service.fetch_universe", exc, {})
            finished = self.now_fn()
            return ScanRunResult(
                started_at=started,
                finished_at=finished,
                btc_context=btc_ctx,
                signals=[],
                prices_by_symbol={},
                symbol_states=symbol_states,
                scanned_symbols=0,
                error_count=error_count,
            )

        candles_by_symbol, fetch_errors = self.fetcher.fetch_many_candles(symbols, self.settings.runtime.worker_count)
        for symbol, message in fetch_errors.items():
            error_count += 1
            if len(error_samples) < 5:
                error_samples.append(f"{symbol}:{message}")
            self.repos.errors.insert(
                ErrorEvent(
                    source="scan_service.fetch_candles",
                    error_type="FetchError",
                    message=message,
                    context={"symbol": symbol},
                )
            )
            self.logger.error("scan_fetch_error", extra={"event": {"symbol": symbol, "msg": message}})

        for symbol, candles in candles_by_symbol.items():
            try:
                eval_result = self.signal_engine.evaluate_detailed(symbol=symbol, candles=candles, btc_context=btc_ctx)
                signal, decisions = eval_result.signal, eval_result.decisions
                symbol_states[symbol] = eval_result.symbol_state
                signal_id = self.repos.signals.insert_signal(signal)
                signal.meta["db_signal_id"] = signal_id
                for d in decisions:
                    d.signal_id = signal_id
                self.repos.signals.insert_decisions(signal_id, decisions)
                signals.append(signal)
                prices[symbol] = signal.price

                if self.settings.runtime.heavy_debug:
                    self.logger.info(
                        "signal_evaluated",
                        extra={
                            "event": {
                                "symbol": signal.symbol,
                                "score": signal.power_score,
                                "candidate": signal.candidate_pass,
                                "auto": signal.auto_pass,
                                "blocked": signal.blocked_reasons,
                                "power_breakdown": signal.power_breakdown,
                            }
                        },
                    )
            except Exception as exc:
                error_count += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{symbol}:{exc.__class__.__name__}")
                self._record_error("scan_service.symbol", exc, {"symbol": symbol})
                continue

        finished = self.now_fn()
        payload = {
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "scanned_symbols": len(signals),
            "errors": error_count,
            "auto_signals": sum(1 for s in signals if s.auto_pass),
        }
        self.repos.runtime_state.set_json("last_scan", payload)
        self.repos.heartbeats.insert(
            component="scanner",
            status="ok" if error_count == 0 else "degraded",
            meta=payload,
        )
        self.logger.info(
            "scan_completed",
            extra={"event": {"scanned": len(signals), "errors": error_count, "auto": payload["auto_signals"]}},
        )
        if self.settings.notifications.notify_on_scan_degraded and error_count > 0:
            sample_text = " | ".join(error_samples[:3]) if error_samples else "-"
            self._notify(
                "rbdcrypt: scan degraded",
                (
                    f"errors={error_count} scanned={len(signals)} auto={payload['auto_signals']}\n"
                    f"samples={sample_text}"
                ),
                priority=4,
                tags="warning",
            )
        if self.settings.notifications.notify_on_auto_signal_summary and payload["auto_signals"] > 0:
            top_n = max(1, int(self.settings.notifications.auto_signal_top_n))
            top = sorted((s for s in signals if s.auto_pass), key=lambda x: x.power_score, reverse=True)[:top_n]
            lines = [self._format_auto_signal_line(i + 1, s) for i, s in enumerate(top)]
            self._notify(
                "rbdcrypt: auto signals",
                "\n".join(lines),
                priority=3,
                tags="chart_with_upwards_trend",
            )
        if self.settings.notifications.notify_on_cycle_summary and self.notification_service is None:
            self._maybe_notify_cycle_summary(started, finished, signals, error_count)
        return ScanRunResult(
            started_at=started,
            finished_at=finished,
            btc_context=btc_ctx,
            signals=signals,
            prices_by_symbol=prices,
            symbol_states=symbol_states,
            scanned_symbols=len(signals),
            error_count=error_count,
        )

    def _record_error(self, source: str, exc: Exception, context: dict[str, object]) -> None:
        tb_single = traceback.format_exc().replace("\n", "\\n")
        self.repos.errors.insert(
            ErrorEvent(
                source=source,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_single_line=tb_single,
                context=context,
            )
        )
        self.logger.error(
            "scan_error",
            extra={"event": {"source": source, "type": exc.__class__.__name__, "msg": str(exc), **context}},
        )
        if self.settings.notifications.notify_on_runtime_error and source not in {"scan_service.fetch_candles", "scan_service.symbol"}:
            symbol_raw = context.get("symbol")
            symbol = str(symbol_raw) if isinstance(symbol_raw, str) and symbol_raw else "-"
            if self.notification_service is not None:
                self.notification_service.on_error(
                    source=source,
                    error=exc,
                    symbol=symbol,
                    pnl_pct=None,
                    active_positions=self.repos.positions.count_active(),
                    scanned_count=0,
                )
            else:
                self._notify(
                    "rbdcrypt: scan error",
                    f"{source} {exc.__class__.__name__}: {exc}",
                    priority=5,
                    tags="rotating_light",
                )

    def _notify(self, title: str, message: str, *, priority: int = 3, tags: str | None = None) -> None:
        if self.notifier is None:
            return
        try:
            self.notifier.notify(title, message, priority=priority, tags=tags)
        except Exception as exc:
            self.logger.error("ntfy_error", extra={"event": {"source": "scan_service", "msg": str(exc)}})

    def _format_auto_signal_line(self, rank: int, signal: SignalEvent) -> str:
        detail = self.settings.notifications.detail_level
        if detail == "compact":
            return f"{rank}) {signal.symbol} {signal.direction.value.upper()} score={signal.power_score:.1f}"
        rsi = signal.metrics.get("rsi")
        adx = signal.metrics.get("adx")
        atr_pct = signal.metrics.get("atr_pct")
        vol_ratio = signal.metrics.get("vol_ratio")
        return (
            f"{rank}) {signal.symbol} {signal.direction.value.upper()} score={signal.power_score:.1f} "
            f"rsi={self._fmt(rsi)} adx={self._fmt(adx)} atr%={self._fmt_pct(atr_pct)} vol={self._fmt(vol_ratio)}"
        )

    def _maybe_notify_cycle_summary(
        self,
        started: datetime,
        finished: datetime,
        signals: list[SignalEvent],
        error_count: int,
    ) -> None:
        now = finished
        min_interval = timedelta(minutes=max(1, int(self.settings.notifications.cycle_summary_min_interval_minutes)))
        if self._last_cycle_summary_notify_at and (now - self._last_cycle_summary_notify_at) < min_interval:
            return
        self._last_cycle_summary_notify_at = now
        auto_signals = [s for s in signals if s.auto_pass]
        directions = Counter(s.direction.value for s in auto_signals)
        msg = (
            f"window={started.isoformat()} -> {finished.isoformat()}\n"
            f"scanned={len(signals)} auto={len(auto_signals)} errors={error_count}\n"
            f"auto_long={directions.get('long', 0)} auto_short={directions.get('short', 0)}"
        )
        self._notify("rbdcrypt: cycle summary", msg, priority=2, tags="information_source")

    @staticmethod
    def _fmt(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}"

    @staticmethod
    def _fmt_pct(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.2f}"
