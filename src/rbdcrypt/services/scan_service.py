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
    chart_points_by_symbol: dict[str, list[float]] = field(default_factory=dict)
    symbol_states: dict[str, SymbolBarState] = field(default_factory=dict)
    scanned_symbols: int = 0
    error_count: int = 0
    outcome_counts: dict[str, int] = field(default_factory=dict)
    blocked_reason_counts: dict[str, int] = field(default_factory=dict)
    score_summary_by_outcome: dict[str, dict[str, float]] = field(default_factory=dict)

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
        chart_points_by_symbol: dict[str, list[float]] = {}
        symbol_states: dict[str, SymbolBarState] = {}
        htf_candles_by_symbol: dict[str, object] = {}
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
                chart_points_by_symbol=chart_points_by_symbol,
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

        htf_bias_enabled, htf_timeframe = self.signal_engine.htf_bias_requirements()
        if htf_bias_enabled:
            htf_limit = max(int(self.settings.binance.kline_limit), 120)
            htf_results, htf_errors = self.fetcher.fetch_many_candles_for_interval(
                symbols=symbols,
                interval=htf_timeframe,
                worker_count=self.settings.runtime.worker_count,
                limit=htf_limit,
            )
            htf_candles_by_symbol = htf_results
            for symbol, message in htf_errors.items():
                error_count += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{symbol}:htf:{message}")
                self.repos.errors.insert(
                    ErrorEvent(
                        source="scan_service.fetch_htf_candles",
                        error_type="FetchError",
                        message=message,
                        context={"symbol": symbol, "timeframe": htf_timeframe},
                    )
                )
                self.logger.error(
                    "scan_fetch_htf_error",
                    extra={"event": {"symbol": symbol, "timeframe": htf_timeframe, "msg": message}},
                )

        for symbol, candles in candles_by_symbol.items():
            try:
                eval_result = self.signal_engine.evaluate_detailed(
                    symbol=symbol,
                    candles=candles,
                    btc_context=btc_ctx,
                    htf_candles=htf_candles_by_symbol.get(symbol),
                )
                signal, decisions = eval_result.signal, eval_result.decisions
                symbol_states[symbol] = eval_result.symbol_state
                signal_id = self.repos.signals.insert_signal(signal)
                signal.meta["db_signal_id"] = signal_id
                for d in decisions:
                    d.signal_id = signal_id
                self.repos.signals.insert_decisions(signal_id, decisions)
                signals.append(signal)
                prices[symbol] = signal.price
                chart_points_by_symbol[symbol] = [float(v) for v in candles.closes[-60:]]

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
                self.logger.info(
                    "signal_outcome",
                    extra={
                        "event": {
                            "symbol": signal.symbol,
                            "outcome": signal.meta.get("evaluator_outcome"),
                            "candidate": signal.candidate_pass,
                            "auto": signal.auto_pass,
                            "blocked_reason_codes": signal.meta.get("blocked_reason_codes", []),
                            "downgrade_reason_codes": signal.meta.get("downgrade_reason_codes", []),
                            "rejection_stage": signal.meta.get("rejection_stage", ""),
                            "power_score": signal.power_score,
                        }
                    },
                )
            except Exception as exc:
                error_count += 1
                if len(error_samples) < 5:
                    error_samples.append(f"{symbol}:{exc.__class__.__name__}")
                self._record_error("scan_service.symbol", exc, {"symbol": symbol})
                continue

        outcome_counts = {"blocked": 0, "candidate": 0, "auto": 0}
        blocked_reason_counts: Counter[str] = Counter()
        scores_by_outcome: dict[str, list[float]] = {"blocked": [], "candidate": [], "auto": []}
        for signal in signals:
            outcome_raw = signal.meta.get("evaluator_outcome")
            outcome = str(outcome_raw).lower() if isinstance(outcome_raw, str) else "blocked"
            if outcome not in outcome_counts:
                outcome = "blocked"
            outcome_counts[outcome] += 1
            scores_by_outcome[outcome].append(float(signal.power_score))
            reason_codes = signal.meta.get("blocked_reason_codes", [])
            if isinstance(reason_codes, list):
                for reason in reason_codes:
                    if isinstance(reason, str) and reason:
                        blocked_reason_counts[reason] += 1

        score_summary_by_outcome: dict[str, dict[str, float]] = {}
        for outcome, values in scores_by_outcome.items():
            if not values:
                score_summary_by_outcome[outcome] = {"min": 0.0, "max": 0.0, "avg": 0.0, "count": 0.0}
                continue
            score_summary_by_outcome[outcome] = {
                "min": float(min(values)),
                "max": float(max(values)),
                "avg": float(sum(values) / max(1, len(values))),
                "count": float(len(values)),
            }

        finished = self.now_fn()
        payload = {
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "scanned_symbols": len(signals),
            "errors": error_count,
            "auto_signals": sum(1 for s in signals if s.auto_pass),
            "outcome_counts": outcome_counts,
            "top_blocked_reasons": blocked_reason_counts.most_common(8),
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
        self.logger.info(
            "scan_diagnostics",
            extra={
                "event": {
                    "outcome_counts": outcome_counts,
                    "top_blocked_reasons": blocked_reason_counts.most_common(8),
                    "score_summary_by_outcome": score_summary_by_outcome,
                }
            },
        )
        if self.settings.notifications.notify_on_scan_degraded and error_count > 0:
            sample_text = " | ".join(error_samples[:3]) if error_samples else "-"
            self._notify(
                "rbdcrypt: scan degraded",
                (
                    f"errors={error_count} scanned={len(signals)} auto={payload['auto_signals']}\n"
                    f"samples={sample_text}"
                ),
                priority=3,
                tags="warning",
            )
        if (
            self.settings.notifications.notify_on_auto_signal_summary
            and self.notification_service is None
            and payload["auto_signals"] > 0
        ):
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
            chart_points_by_symbol=chart_points_by_symbol,
            symbol_states=symbol_states,
            scanned_symbols=len(signals),
            error_count=error_count,
            outcome_counts=outcome_counts,
            blocked_reason_counts=dict(blocked_reason_counts),
            score_summary_by_outcome=score_summary_by_outcome,
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
                    priority=3,
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
        self._notify("rbdcrypt: cycle summary", msg, priority=3, tags="information_source")

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
