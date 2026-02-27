from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import AppSettings
from ..models.market_context import MarketContext
from ..models.signal import SignalDecision, SignalDirection, SignalEvent
from ..models.symbol_state import SymbolBarState
from .legacy_parity import (
    LegacyAnalysis,
    analyze_candles,
    btc_trend_match,
    calculate_power_score,
    derive_btc_context_from_analysis,
    evaluate_signal_filters,
    get_candidate_fail_reason,
    get_flip_candidate_signal,
    get_signal_thresholds,
    score_from_flags,
)
from .signal_engine import CandleSeries


@dataclass(slots=True)
class SignalEvaluationResult:
    signal: SignalEvent
    decisions: list[SignalDecision]
    symbol_state: SymbolBarState
    analysis: LegacyAnalysis


class ParitySignalEngine:
    def __init__(self, settings: AppSettings, interval: str = "5m") -> None:
        self.settings = settings
        self.interval = interval

    def derive_btc_market_context(self, candles: CandleSeries) -> tuple[MarketContext, SymbolBarState, LegacyAnalysis]:
        analysis = analyze_candles(candles, self.settings.legacy_parity)
        analysis.symbol_state.symbol = self.settings.binance.btc_symbol
        ctx = derive_btc_context_from_analysis(
            analysis,
            interval=self.interval,
            symbol=self.settings.binance.btc_symbol,
            cfg=self.settings.legacy_parity,
        )
        return ctx, analysis.symbol_state, analysis

    def evaluate(self, *, symbol: str, candles: CandleSeries, btc_context: MarketContext | None) -> tuple[SignalEvent, list[SignalDecision]]:
        result = self.evaluate_detailed(symbol=symbol, candles=candles, btc_context=btc_context)
        return result.signal, result.decisions

    def evaluate_detailed(
        self,
        *,
        symbol: str,
        candles: CandleSeries,
        btc_context: MarketContext | None,
    ) -> SignalEvaluationResult:
        cfg = self.settings.legacy_parity
        analysis = analyze_candles(candles, cfg)
        analysis.symbol_state.symbol = symbol
        r = analysis.closed_row

        candidate_signal = get_flip_candidate_signal(r)
        r_eval = r
        if candidate_signal is None and cfg.allow_trend_continuation_entry:
            trend = int(r.get("TREND", 0))
            if trend > 0:
                candidate_signal = "LONG"
            elif trend < 0:
                candidate_signal = "SHORT"
            if candidate_signal is not None:
                # Continuation mode synthesizes a directional trigger when no explicit flip exists.
                r_eval = dict(r)
                r_eval["FLIP_LONG"] = candidate_signal == "LONG"
                r_eval["FLIP_SHORT"] = candidate_signal == "SHORT"
        cand_t = get_signal_thresholds(cfg, "candidate")
        auto_t = get_signal_thresholds(cfg, "auto")
        candidate_flags = evaluate_signal_filters(r_eval, candidate_signal, cand_t)
        auto_flags = evaluate_signal_filters(r_eval, candidate_signal, auto_t)
        candidate_power, candidate_breakdown = calculate_power_score(r_eval, cand_t)
        auto_power, auto_breakdown = calculate_power_score(r_eval, auto_t)
        candidate_score = score_from_flags(candidate_flags)
        auto_score = score_from_flags(auto_flags)

        blocked_reason: str | None = None
        rejection_stage = ""
        effective_auto_power = auto_power
        btc_match = False
        is_chop_market = False

        candidate_pass = False
        auto_pass = False

        if not candidate_signal:
            blocked_reason = "NO_FLIP"
            rejection_stage = "candidate_filter"
        elif not candidate_flags["all_ok"]:
            blocked_reason = get_candidate_fail_reason(r, candidate_signal, cfg)
            rejection_stage = "candidate_filter"
        elif candidate_power < cand_t.min_power_score:
            blocked_reason = f"CAND_LOW_POWER_{candidate_power:.0f}"
            rejection_stage = "candidate_filter"
        else:
            candidate_pass = True
            if not auto_flags["all_ok"]:
                blocked_reason = "AUTO_TECH_FAIL"
                rejection_stage = "auto_filter"
            else:
                if btc_context is not None:
                    btc_match = btc_trend_match(candidate_signal, btc_context)
                    btc_trend_code = int(btc_context.meta.get("trend_code", 0)) if isinstance(btc_context.meta, dict) else 0
                    is_chop_market = bool(btc_context.meta.get("is_chop_market", False)) if isinstance(btc_context.meta, dict) else False
                    if cfg.auto_btc_trend_mode == "soft_penalty":
                        if not btc_match:
                            effective_auto_power -= float(cfg.auto_btc_trend_penalty)
                    else:
                        if btc_trend_code == 0:
                            blocked_reason = "BTC_VERI_YOK"
                        elif not btc_match:
                            blocked_reason = "BTC_TREND_KOTU"
                        if blocked_reason:
                            rejection_stage = "market_filter"
                else:
                    if cfg.auto_btc_trend_mode == "hard_block":
                        blocked_reason = "BTC_VERI_YOK"
                        rejection_stage = "market_filter"
                    else:
                        effective_auto_power -= float(cfg.auto_btc_trend_penalty)
                        btc_match = False
                if not blocked_reason and is_chop_market:
                    if cfg.auto_chop_policy == "block":
                        blocked_reason = "CHOP_MARKET"
                        rejection_stage = "market_filter"
                    elif cfg.auto_chop_policy == "penalty":
                        effective_auto_power -= float(cfg.auto_chop_penalty)
                if not blocked_reason and effective_auto_power < auto_t.min_power_score:
                    blocked_reason = f"LOW_POWER_{effective_auto_power:.0f}"
                    rejection_stage = "auto_filter"
                if not blocked_reason and float(r["ADX"]) < float(cfg.chop_adx_threshold):
                    blocked_reason = "LOW_ADX"
                    rejection_stage = "execution_filter"
                if not blocked_reason:
                    auto_pass = True

        direction = SignalDirection.FLAT
        if candidate_signal == "LONG":
            direction = SignalDirection.LONG
        elif candidate_signal == "SHORT":
            direction = SignalDirection.SHORT

        event_score = effective_auto_power if (candidate_signal and auto_flags["all_ok"]) else candidate_power
        blocked_reasons = [blocked_reason] if blocked_reason else []
        signal = SignalEvent(
            symbol=symbol,
            interval=self.interval,
            bar_time=r["timestamp"],  # type: ignore[arg-type]
            direction=direction,
            price=float(r["close"]),
            power_score=float(round(event_score, 2)),
            metrics={
                "rsi": float(r["RSI"]),
                "adx": float(r["ADX"]),
                "atr": float(r["ATR_14"]),
                "atr_pct": float(r["ATR_PCT"]),
                "ema20": float(r["EMA20"]),
                "ema50": float(r["EMA50"]),
                "vol_ratio": float(r["VOL_RATIO"]),
                "macd_hist": float(r["MACD_HIST"]),
                "trend": float(r["TREND"]),
                "candidate_score": float(candidate_score),
                "auto_score": float(auto_score),
                "candidate_power": float(candidate_power),
                "auto_power": float(auto_power),
                "effective_auto_power": float(effective_auto_power),
            },
            power_breakdown={
                **{f"candidate_{k}": v for k, v in candidate_breakdown.items()},
                **{f"auto_{k}": v for k, v in auto_breakdown.items()},
                "penalty_btc": -float(cfg.auto_btc_trend_penalty) if (candidate_signal and auto_flags["all_ok"] and not btc_match and cfg.auto_btc_trend_mode == "soft_penalty") else 0.0,
                "penalty_chop": -float(cfg.auto_chop_penalty) if (candidate_signal and auto_flags["all_ok"] and is_chop_market and cfg.auto_chop_policy == "penalty") else 0.0,
            },
            candidate_pass=candidate_pass,
            auto_pass=auto_pass,
            blocked_reasons=blocked_reasons,
            meta={
                "candidate_signal": candidate_signal,
                "candidate_flags": candidate_flags,
                "auto_flags": auto_flags,
                "candidate_score": candidate_score,
                "auto_score": auto_score,
                "rejection_stage": rejection_stage,
                "blocked_reason": blocked_reason,
                "btc_trend_match": btc_match,
                "is_chop_market": is_chop_market,
                "entry_atr14": float(r["ATR_14"]),
                "closed_trend": int(r["TREND"]),
            },
        )

        decisions = [
            SignalDecision(
                symbol=symbol,
                bar_time=signal.bar_time,
                stage="candidate_filter",
                outcome="pass" if candidate_pass else "blocked",
                blocked_reason=None if candidate_pass else (blocked_reason or "NO_FLIP"),
                decision_payload={
                    "candidate_signal": candidate_signal,
                    "flags": candidate_flags,
                    "thresholds": asdict(cand_t),
                    "candidate_power": candidate_power,
                    "candidate_score": candidate_score,
                },
            ),
            SignalDecision(
                symbol=symbol,
                bar_time=signal.bar_time,
                stage="auto_filter_market",
                outcome="auto_pass" if auto_pass else "blocked",
                blocked_reason=None if auto_pass else blocked_reason,
                decision_payload={
                    "flags": auto_flags,
                    "thresholds": asdict(auto_t),
                    "auto_power": auto_power,
                    "effective_auto_power": effective_auto_power,
                    "auto_score": auto_score,
                    "btc_trend_match": btc_match,
                    "btc_mode": cfg.auto_btc_trend_mode,
                    "is_chop_market": is_chop_market,
                    "chop_policy": cfg.auto_chop_policy,
                    "rejection_stage": rejection_stage,
                },
            ),
        ]
        return SignalEvaluationResult(signal=signal, decisions=decisions, symbol_state=analysis.symbol_state, analysis=analysis)
