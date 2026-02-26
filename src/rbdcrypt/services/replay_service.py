from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta

from ..brokers.paper_broker import PaperBroker
from ..config import AppSettings
from ..core.risk import build_risk_plan_from_levels
from ..core.state_machine import update_position_mark
from ..models.ohlcv import OHLCVBar
from ..storage.repositories import Repositories
from ..strategy.exit_engine import evaluate_legacy_exit
from ..strategy.parity_signal_engine import ParitySignalEngine
from ..strategy.signal_engine import CandleSeries


@dataclass(slots=True)
class ReplayTradeRecord:
    symbol: str
    side: str
    opened_at: str
    closed_at: str
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_pct: float
    pnl_quote: float
    rr_initial: float


@dataclass(slots=True)
class ReplayReport:
    symbol: str
    interval: str
    start: str | None
    end: str | None
    bars_used: int
    bars_skipped_unaligned: int
    candidate_signals: int = 0
    auto_signals: int = 0
    opened: int = 0
    closed: int = 0
    missed: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl_quote: float = 0.0
    cooldown_blocks: int = 0
    trades: list[ReplayTradeRecord] = field(default_factory=list)


class ReplayService:
    def __init__(self, *, settings: AppSettings, repos: Repositories, signal_engine: ParitySignalEngine, logger) -> None:
        self.settings = settings
        self.repos = repos
        self.signal_engine = signal_engine
        self.logger = logger
        self._broker = PaperBroker()

    def replay_symbol(
        self,
        *,
        symbol: str,
        interval: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        warmup_bars: int = 80,
        max_trades: int | None = None,
        persist_report: bool = False,
    ) -> ReplayReport:
        interval = interval or self.settings.binance.interval
        target_bars = self.repos.candles.list_bars(symbol=symbol, interval=interval, start=start, end=end, ascending=True)
        btc_bars = self.repos.candles.list_bars(
            symbol=self.settings.binance.btc_symbol,
            interval=interval,
            start=start,
            end=end,
            ascending=True,
        )
        if not target_bars:
            raise ValueError(f"No backfilled candles found for {symbol} ({interval})")
        if not btc_bars:
            raise ValueError(f"No backfilled candles found for {self.settings.binance.btc_symbol} ({interval})")

        btc_map = {b.open_time: b for b in btc_bars}
        aligned_target: list[OHLCVBar] = []
        aligned_btc: list[OHLCVBar] = []
        skipped_unaligned = 0
        for b in target_bars:
            btc = btc_map.get(b.open_time)
            if btc is None:
                skipped_unaligned += 1
                continue
            aligned_target.append(b)
            aligned_btc.append(btc)
        if len(aligned_target) < max(warmup_bars, 60):
            raise ValueError("Insufficient aligned bars for replay")

        report = ReplayReport(
            symbol=symbol,
            interval=interval,
            start=start.isoformat() if start else None,
            end=end.isoformat() if end else None,
            bars_used=len(aligned_target),
            bars_skipped_unaligned=skipped_unaligned,
        )

        active_position = None
        cooldown_until: datetime | None = None
        balance = float(self.settings.balance.starting_balance)
        for i in range(max(warmup_bars, 60), len(aligned_target)):
            tgt_slice = aligned_target[: i + 1]
            btc_slice = aligned_btc[: i + 1]
            tgt_series = self._bars_to_series(tgt_slice)
            btc_series = self._bars_to_series(btc_slice)
            btc_ctx, _, _ = self.signal_engine.derive_btc_market_context(btc_series)
            eval_result = self.signal_engine.evaluate_detailed(symbol=symbol, candles=tgt_series, btc_context=btc_ctx)
            signal = eval_result.signal
            state = eval_result.symbol_state
            now = state.current_bar_time

            if active_position is not None:
                update_position_mark(active_position, state.current_close, now)
                decision = evaluate_legacy_exit(
                    position=active_position,
                    current_high=state.current_high,
                    current_low=state.current_low,
                    current_close=state.current_close,
                    current_trend=state.current_trend,
                    current_ema20=state.current_ema20,
                    now=now,
                    legacy_cfg=self.settings.legacy_parity,
                )
                if decision.should_exit and decision.reason and decision.exit_price is not None:
                    trade = self._broker.close_position(
                        position=active_position,
                        exit_price=decision.exit_price,
                        reason=decision.reason,
                        closed_at=now,
                        fee_pct_per_side=self.settings.risk.fee_pct_per_side,
                    )
                    report.closed += 1
                    report.total_pnl_quote += trade.pnl_quote
                    if trade.pnl_quote >= 0:
                        report.wins += 1
                    else:
                        report.losses += 1
                    report.trades.append(
                        ReplayTradeRecord(
                            symbol=trade.symbol,
                            side=trade.side.value,
                            opened_at=trade.opened_at.isoformat(),
                            closed_at=trade.closed_at.isoformat(),
                            entry_price=trade.entry_price,
                            exit_price=trade.exit_price,
                            exit_reason=trade.exit_reason,
                            pnl_pct=trade.pnl_pct,
                            pnl_quote=trade.pnl_quote,
                            rr_initial=trade.rr_initial,
                        )
                    )
                    balance += trade.pnl_quote
                    cooldown_until = now + timedelta(minutes=self.settings.legacy_parity.cooldown_minutes)
                    active_position = None
                    if max_trades is not None and report.closed >= max_trades:
                        break

            if signal.candidate_pass:
                report.candidate_signals += 1
            if signal.auto_pass:
                report.auto_signals += 1

            if active_position is not None:
                if signal.auto_pass:
                    report.missed += 1
                continue
            if not signal.auto_pass or signal.direction.value == "flat":
                continue
            if cooldown_until is not None and now < cooldown_until:
                report.missed += 1
                report.cooldown_blocks += 1
                continue
            atr_val = float(signal.meta.get("entry_atr14", 0.0) or 0.0)
            if atr_val <= 0.0:
                report.missed += 1
                continue
            if signal.direction.value == "long":
                initial_sl = signal.price - (self.settings.legacy_parity.sl_atr_mult * atr_val)
                initial_tp = signal.price + (self.settings.legacy_parity.tp_atr_mult * atr_val)
            else:
                initial_sl = signal.price + (self.settings.legacy_parity.sl_atr_mult * atr_val)
                initial_tp = signal.price - (self.settings.legacy_parity.tp_atr_mult * atr_val)
            risk_plan = build_risk_plan_from_levels(
                balance=balance,
                risk_per_trade_pct=self.settings.risk.risk_per_trade_pct,
                leverage=self.settings.risk.leverage,
                entry_price=signal.price,
                initial_sl=initial_sl,
                initial_tp=initial_tp,
                side=signal.direction.value,
                min_notional=self.settings.risk.min_notional,
            )
            if risk_plan.rr_initial < self.settings.risk.min_rr:
                report.missed += 1
                continue
            active_position = self._broker.open_position(
                symbol=symbol,
                side=signal.direction.value,
                risk_plan=risk_plan,
                opened_at=now,
                entry_bar_time=signal.bar_time,
                strategy_tag="replay",
            )
            active_position.leverage = self.settings.risk.leverage
            report.opened += 1

        payload = {
            "symbol": report.symbol,
            "interval": report.interval,
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
            "trades": [asdict(t) for t in report.trades[-25:]],
        }
        if persist_report:
            self.repos.runtime_state.set_json(f"last_replay:{symbol}:{interval}", payload)
        self.logger.info(
            "replay_completed",
            extra={
                "event": {
                    "symbol": symbol,
                    "interval": interval,
                    "bars": report.bars_used,
                    "opened": report.opened,
                    "closed": report.closed,
                    "pnl_quote": round(report.total_pnl_quote, 4),
                }
            },
        )
        return report

    @staticmethod
    def _bars_to_series(bars: list[OHLCVBar]) -> CandleSeries:
        return CandleSeries(
            open_times=[b.open_time for b in bars],
            opens=[b.open for b in bars],
            highs=[b.high for b in bars],
            lows=[b.low for b in bars],
            closes=[b.close for b in bars],
            volumes=[b.volume for b in bars],
        )
