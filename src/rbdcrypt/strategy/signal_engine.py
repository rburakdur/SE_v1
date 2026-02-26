from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from ..config import FilterSettings, ScoreSettings
from ..indicators.talib_indicators import latest_snapshot
from ..models.market_context import MarketContext
from ..models.signal import SignalDecision, SignalDirection, SignalEvent
from .filters import FilterInputs, evaluate_filters
from .scoring import compute_power_score


@dataclass(slots=True)
class CandleSeries:
    open_times: Sequence[datetime]
    opens: Sequence[float]
    highs: Sequence[float]
    lows: Sequence[float]
    closes: Sequence[float]
    volumes: Sequence[float]

    def validate(self) -> None:
        n = len(self.closes)
        if n < 60:
            raise ValueError("Need at least 60 candles for signal evaluation")
        if not (len(self.highs) == len(self.lows) == len(self.opens) == len(self.volumes) == n):
            raise ValueError("CandleSeries arrays must have same length")
        if len(self.open_times) != n:
            raise ValueError("CandleSeries times length mismatch")


class SignalEngine:
    def __init__(self, filter_cfg: FilterSettings, score_cfg: ScoreSettings, interval: str = "5m") -> None:
        self.filter_cfg = filter_cfg
        self.score_cfg = score_cfg
        self.interval = interval

    def evaluate(
        self,
        *,
        symbol: str,
        candles: CandleSeries,
        btc_context: MarketContext | None,
    ) -> tuple[SignalEvent, list[SignalDecision]]:
        candles.validate()
        snap = latest_snapshot(
            close=candles.closes,
            high=candles.highs,
            low=candles.lows,
            volume=candles.volumes,
            ema_fast_period=self.filter_cfg.ema_fast_period,
            ema_slow_period=self.filter_cfg.ema_slow_period,
        )
        price = float(candles.closes[-1])
        bar_time = candles.open_times[-1]
        inputs = FilterInputs(
            price=price,
            ema_fast=snap.ema_fast,
            ema_slow=snap.ema_slow,
            rsi=snap.rsi,
            adx=snap.adx,
            atr_pct=snap.atr_pct,
            volume_ratio=snap.volume_ratio,
        )
        outcome = evaluate_filters(inputs, self.filter_cfg, btc_context)
        power = compute_power_score(inputs, outcome, self.score_cfg)

        candidate_pass = (
            power.total >= self.score_cfg.candidate_score_min
            and outcome.direction != "flat"
            and len(outcome.blocked_reasons) == 0
        )
        auto_pass = power.total >= self.score_cfg.auto_score_min and candidate_pass

        signal = SignalEvent(
            symbol=symbol,
            interval=self.interval,
            bar_time=bar_time,
            direction=SignalDirection(outcome.direction),
            price=price,
            power_score=power.total,
            metrics={
                "ema_fast": snap.ema_fast,
                "ema_slow": snap.ema_slow,
                "rsi": snap.rsi,
                "adx": snap.adx,
                "atr": snap.atr,
                "atr_pct": snap.atr_pct,
                "volume_ratio": snap.volume_ratio,
            },
            power_breakdown=power.breakdown,
            candidate_pass=candidate_pass,
            auto_pass=auto_pass,
            blocked_reasons=list(outcome.blocked_reasons),
            meta={
                "btc_aligned": outcome.btc_aligned,
                "is_choppy": outcome.is_choppy,
            },
        )
        decisions = [
            SignalDecision(
                symbol=symbol,
                bar_time=bar_time,
                stage="filters",
                outcome="pass" if len(outcome.blocked_reasons) == 0 else "blocked",
                blocked_reason=",".join(outcome.blocked_reasons) if outcome.blocked_reasons else None,
                decision_payload={
                    "direction": outcome.direction,
                    "is_choppy": outcome.is_choppy,
                    "btc_aligned": outcome.btc_aligned,
                },
            ),
            SignalDecision(
                symbol=symbol,
                bar_time=bar_time,
                stage="scoring",
                outcome="candidate" if candidate_pass else "reject",
                blocked_reason=None if candidate_pass else "candidate_threshold",
                decision_payload={
                    "power_score": power.total,
                    "candidate_min": self.score_cfg.candidate_score_min,
                    "auto_min": self.score_cfg.auto_score_min,
                    "breakdown": power.breakdown,
                },
            ),
            SignalDecision(
                symbol=symbol,
                bar_time=bar_time,
                stage="auto_trade",
                outcome="auto_pass" if auto_pass else "manual_only",
                blocked_reason=None if auto_pass else "auto_threshold",
                decision_payload={"power_score": power.total},
            ),
        ]
        return signal, decisions


def derive_btc_market_context(
    *,
    candles: CandleSeries,
    filter_cfg: FilterSettings,
    interval: str = "5m",
    symbol: str = "BTCUSDT",
) -> MarketContext:
    candles.validate()
    snap = latest_snapshot(
        close=candles.closes,
        high=candles.highs,
        low=candles.lows,
        volume=candles.volumes,
        ema_fast_period=filter_cfg.ema_fast_period,
        ema_slow_period=filter_cfg.ema_slow_period,
    )
    if snap.ema_fast > snap.ema_slow and snap.rsi >= 50.0:
        trend_direction = "up"
    elif snap.ema_fast < snap.ema_slow and snap.rsi <= 50.0:
        trend_direction = "down"
    else:
        trend_direction = "flat"
    chop_state = "choppy" if (snap.atr_pct <= filter_cfg.atr_pct_max_chop and snap.adx < filter_cfg.adx_min + 5) else "trending"
    trend_score = 0.0
    if trend_direction == "up":
        trend_score = min(100.0, 50.0 + (snap.rsi - 50.0) + snap.adx)
    elif trend_direction == "down":
        trend_score = min(100.0, 50.0 + (50.0 - snap.rsi) + snap.adx)
    return MarketContext(
        symbol=symbol,
        interval=interval,
        bar_time=candles.open_times[-1],
        trend_direction=trend_direction,  # type: ignore[arg-type]
        trend_score=round(trend_score, 4),
        chop_state=chop_state,  # type: ignore[arg-type]
        metrics={
            "ema_fast": snap.ema_fast,
            "ema_slow": snap.ema_slow,
            "rsi": snap.rsi,
            "adx": snap.adx,
            "atr_pct": snap.atr_pct,
            "volume_ratio": snap.volume_ratio,
        },
    )
