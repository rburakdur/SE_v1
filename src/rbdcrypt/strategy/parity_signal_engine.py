from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Literal

import numpy as np

from ..config import AppSettings
from ..indicators.talib_indicators import ema, ensure_float64_contiguous
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
from .profile_config import StrategyProfile
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
        self.strategy_profile: StrategyProfile = self.settings.load_strategy_profile()

    def htf_bias_requirements(self) -> tuple[bool, str]:
        cfg = self.strategy_profile.filters.htf_bias
        return bool(cfg.directional_filter), str(cfg.timeframe)

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

    def evaluate(
        self,
        *,
        symbol: str,
        candles: CandleSeries,
        btc_context: MarketContext | None,
        htf_candles: CandleSeries | None = None,
    ) -> tuple[SignalEvent, list[SignalDecision]]:
        result = self.evaluate_detailed(symbol=symbol, candles=candles, btc_context=btc_context, htf_candles=htf_candles)
        return result.signal, result.decisions

    def evaluate_detailed(
        self,
        *,
        symbol: str,
        candles: CandleSeries,
        btc_context: MarketContext | None,
        htf_candles: CandleSeries | None = None,
    ) -> SignalEvaluationResult:
        cfg = self.settings.legacy_parity
        profile = self.strategy_profile

        analysis = analyze_candles(candles, cfg)
        analysis.symbol_state.symbol = symbol
        r = analysis.closed_row

        candidate_signal, trigger_info = self._resolve_candidate_signal(
            row=r,
            candles=candles,
            allow_trend_continuation=bool(cfg.allow_trend_continuation_entry),
            trigger_mode=profile.filters.ltf_trigger,
        )
        r_eval = dict(r)
        if candidate_signal is not None and bool(trigger_info.get("synthetic_flip", False)):
            # Synthetic signal modes (continuation / hma_cross) emulate flip flags for legacy filter scoring.
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

        candidate_min = float(profile.filters.candidate_min)
        auto_min = float(profile.filters.auto_min)

        htf_bias_direction, htf_bias_info = self._resolve_htf_bias(htf_candles=htf_candles)
        htf_bias_aligned = self._is_bias_aligned(candidate_signal, htf_bias_direction)
        session_pass, session_reason = self._session_filter_pass(
            timestamp=r.get("timestamp"),
            profile=profile,
        )
        is_chop_market = self._is_choppy_market(row=r, btc_context=btc_context, cfg=cfg)

        blocked_reason_codes: list[str] = []
        downgrade_reason_codes: list[str] = []
        rejection_stage = ""
        effective_auto_power = float(auto_power)
        penalty_btc = 0.0
        penalty_chop = 0.0
        btc_match = False

        candidate_pass = False
        auto_pass = False
        evaluator_outcome: Literal["blocked", "candidate", "auto"] = "blocked"

        if not candidate_signal:
            reason_code = str(trigger_info.get("blocked_reason") or "NO_FLIP")
            blocked_reason_codes.append(reason_code)
            rejection_stage = str(trigger_info.get("blocked_stage") or "candidate_trigger")
        elif not bool(trigger_info.get("trigger_pass", True)):
            blocked_reason_codes.append(str(trigger_info.get("blocked_reason") or "TRIGGER_MODE_BLOCKED"))
            rejection_stage = str(trigger_info.get("blocked_stage") or "candidate_trigger")
        elif not candidate_flags["all_ok"]:
            blocked_reason_codes.append(get_candidate_fail_reason(r, candidate_signal, cfg))
            rejection_stage = "candidate_filter"
        elif not session_pass:
            blocked_reason_codes.append(session_reason or "SESSION_FILTER_BLOCKED")
            rejection_stage = "session_filter"
        elif profile.filters.choppiness_filter and is_chop_market:
            blocked_reason_codes.append("CHOPPINESS_FILTER_BLOCKED")
            rejection_stage = "choppiness_filter"
        elif profile.filters.htf_bias.directional_filter and htf_bias_info.get("status") != "ok":
            blocked_reason_codes.append("HTF_BIAS_UNAVAILABLE")
            rejection_stage = "htf_bias_filter"
        elif profile.filters.htf_bias.directional_filter and htf_bias_direction == "flat":
            blocked_reason_codes.append("HTF_BIAS_FLAT")
            rejection_stage = "htf_bias_filter"
        elif profile.filters.htf_bias.directional_filter and not htf_bias_aligned:
            blocked_reason_codes.append("HTF_BIAS_MISMATCH")
            rejection_stage = "htf_bias_filter"
        elif candidate_power < candidate_min:
            blocked_reason_codes.append("CANDIDATE_SCORE_BELOW_MIN")
            rejection_stage = "candidate_score"
        else:
            candidate_pass = True
            evaluator_outcome = "candidate"

            if not auto_flags["all_ok"]:
                downgrade_reason_codes.append("AUTO_TECH_FAIL")
            else:
                if btc_context is not None:
                    btc_match = btc_trend_match(candidate_signal, btc_context)
                    btc_trend_code = int(btc_context.meta.get("trend_code", 0)) if isinstance(btc_context.meta, dict) else 0

                    if cfg.auto_btc_trend_mode == "soft_penalty":
                        if not btc_match:
                            penalty_btc = -float(cfg.auto_btc_trend_penalty)
                            effective_auto_power += penalty_btc
                    else:
                        if btc_trend_code == 0:
                            downgrade_reason_codes.append("BTC_CONTEXT_UNAVAILABLE")
                        elif not btc_match:
                            downgrade_reason_codes.append("BTC_TREND_MISMATCH")
                else:
                    if cfg.auto_btc_trend_mode == "hard_block":
                        downgrade_reason_codes.append("BTC_CONTEXT_UNAVAILABLE")
                    else:
                        penalty_btc = -float(cfg.auto_btc_trend_penalty)
                        effective_auto_power += penalty_btc

                if is_chop_market:
                    if cfg.auto_chop_policy == "block":
                        downgrade_reason_codes.append("CHOP_MARKET")
                    elif cfg.auto_chop_policy == "penalty":
                        penalty_chop = -float(cfg.auto_chop_penalty)
                        effective_auto_power += penalty_chop

                if effective_auto_power < auto_min:
                    downgrade_reason_codes.append("AUTO_SCORE_BELOW_MIN")
                if float(r["ADX"]) < float(cfg.chop_adx_threshold):
                    downgrade_reason_codes.append("AUTO_LOW_ADX")

                if not downgrade_reason_codes:
                    auto_pass = True
                    evaluator_outcome = "auto"
                else:
                    rejection_stage = "auto_filter"

        if evaluator_outcome == "blocked" and not rejection_stage:
            rejection_stage = "candidate_filter"

        direction = SignalDirection.FLAT
        if candidate_signal == "LONG":
            direction = SignalDirection.LONG
        elif candidate_signal == "SHORT":
            direction = SignalDirection.SHORT

        event_score = effective_auto_power if auto_pass else candidate_power
        blocked_reasons = blocked_reason_codes[:] if evaluator_outcome == "blocked" else []
        blocked_reason = blocked_reason_codes[0] if blocked_reason_codes else None
        downgrade_reason = downgrade_reason_codes[0] if downgrade_reason_codes else None

        filter_breakdown = {
            "session_filter": {
                "enabled": bool(profile.filters.session_filter),
                "passed": bool(session_pass),
                "reason": session_reason,
                "no_entry_start_hour_utc": profile.filters.session_no_entry_start_hour_utc,
                "no_entry_end_hour_utc": profile.filters.session_no_entry_end_hour_utc,
            },
            "ltf_trigger": {
                "mode": profile.filters.ltf_trigger,
                "signal": candidate_signal,
                "flip_long": bool(r_eval.get("FLIP_LONG", False)),
                "flip_short": bool(r_eval.get("FLIP_SHORT", False)),
                **trigger_info,
            },
            "candidate_flags": candidate_flags,
            "auto_flags": auto_flags,
            "htf_bias": {
                **htf_bias_info,
                "aligned_with_signal": htf_bias_aligned if candidate_signal is not None else None,
            },
            "market": {
                "btc_trend_match": btc_match,
                "is_chop_market": is_chop_market,
                "auto_btc_trend_mode": cfg.auto_btc_trend_mode,
                "auto_chop_policy": cfg.auto_chop_policy,
            },
        }

        score_breakdown = {
            "candidate": {
                "score": float(candidate_score),
                "power": float(candidate_power),
                "min_required": candidate_min,
                "legacy_min_power": float(cand_t.min_power_score),
                "passed": bool(candidate_power >= candidate_min),
            },
            "auto": {
                "score": float(auto_score),
                "power": float(auto_power),
                "effective_power": float(effective_auto_power),
                "min_required": auto_min,
                "legacy_min_power": float(auto_t.min_power_score),
                "passed": bool(effective_auto_power >= auto_min),
            },
            "penalties": {
                "btc": float(penalty_btc),
                "chop": float(penalty_chop),
            },
        }

        htf_meta_key = f"{profile.filters.htf_bias.ma_type.lower()}_{profile.filters.htf_bias.timeframe}_bias"
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
                "penalty_btc": float(penalty_btc),
                "penalty_chop": float(penalty_chop),
                "candidate_min": candidate_min,
                "auto_min": auto_min,
            },
            candidate_pass=candidate_pass,
            auto_pass=auto_pass,
            blocked_reasons=blocked_reasons,
            meta={
                "strategy_profile": profile.name,
                "profile_id": profile.name,
                "evaluator_outcome": evaluator_outcome,
                "blocked_reason_codes": blocked_reason_codes,
                "downgrade_reason_codes": downgrade_reason_codes,
                "candidate_signal": candidate_signal,
                "trigger_mode": profile.filters.ltf_trigger,
                "bias_mode": f"{profile.filters.htf_bias.ma_type}_{profile.filters.htf_bias.timeframe}",
                "filter_breakdown": filter_breakdown,
                "score_breakdown": score_breakdown,
                "candidate_flags": candidate_flags,
                "auto_flags": auto_flags,
                "candidate_score": candidate_score,
                "auto_score": auto_score,
                "rejection_stage": rejection_stage,
                "blocked_reason": blocked_reason,
                "downgrade_reason": downgrade_reason,
                "btc_trend_match": btc_match,
                "is_chop_market": is_chop_market,
                "entry_atr14": float(r["ATR_14"]),
                "closed_trend": int(r["TREND"]),
                "htf_bias_direction": htf_bias_direction,
                htf_meta_key: htf_bias_direction,
            },
        )

        decisions = [
            SignalDecision(
                symbol=symbol,
                bar_time=signal.bar_time,
                stage="signal_evaluator",
                outcome=evaluator_outcome,
                blocked_reason=blocked_reason,
                decision_payload={
                    "strategy_profile": profile.name,
                    "candidate_signal": candidate_signal,
                    "blocked_reason_codes": blocked_reason_codes,
                    "downgrade_reason_codes": downgrade_reason_codes,
                    "filter_breakdown": filter_breakdown,
                    "score_breakdown": score_breakdown,
                    "trigger_info": trigger_info,
                    "rejection_stage": rejection_stage,
                },
            ),
            SignalDecision(
                symbol=symbol,
                bar_time=signal.bar_time,
                stage="candidate_filter",
                outcome="pass" if candidate_pass else "blocked",
                blocked_reason=None if candidate_pass else blocked_reason,
                decision_payload={
                    "candidate_signal": candidate_signal,
                    "flags": candidate_flags,
                    "thresholds": asdict(cand_t),
                    "candidate_power": candidate_power,
                    "candidate_score": candidate_score,
                    "candidate_min": candidate_min,
                    "session_reason": session_reason,
                    "trigger_info": trigger_info,
                    "rejection_stage": rejection_stage,
                },
            ),
            SignalDecision(
                symbol=symbol,
                bar_time=signal.bar_time,
                stage="auto_filter_market",
                outcome="auto" if auto_pass else ("candidate" if candidate_pass else "blocked"),
                blocked_reason=None if auto_pass else (downgrade_reason if candidate_pass else blocked_reason),
                decision_payload={
                    "flags": auto_flags,
                    "thresholds": asdict(auto_t),
                    "auto_power": auto_power,
                    "effective_auto_power": effective_auto_power,
                    "auto_score": auto_score,
                    "auto_min": auto_min,
                    "btc_trend_match": btc_match,
                    "btc_mode": cfg.auto_btc_trend_mode,
                    "is_chop_market": is_chop_market,
                    "chop_policy": cfg.auto_chop_policy,
                    "rejection_stage": rejection_stage,
                    "downgrade_reason_codes": downgrade_reason_codes,
                },
            ),
        ]
        return SignalEvaluationResult(signal=signal, decisions=decisions, symbol_state=analysis.symbol_state, analysis=analysis)

    @staticmethod
    def _session_filter_pass(
        *,
        timestamp: object,
        profile: StrategyProfile,
    ) -> tuple[bool, str | None]:
        if not profile.filters.session_filter:
            return True, None
        if not isinstance(timestamp, datetime):
            return True, None
        start = profile.filters.session_no_entry_start_hour_utc
        end = profile.filters.session_no_entry_end_hour_utc
        if start is None or end is None:
            return True, None
        ts_utc = timestamp.astimezone(UTC) if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
        hour = int(ts_utc.hour)
        blocked = ParitySignalEngine._hour_in_window(hour=hour, start=start, end=end)
        if blocked:
            reason = f"SESSION_NO_ENTRY_WINDOW_{start:02d}_{end:02d}_UTC"
            return False, reason
        return True, None

    @staticmethod
    def _is_bias_aligned(candidate_signal: str | None, bias: Literal["up", "down", "flat"]) -> bool:
        if candidate_signal == "LONG":
            return bias == "up"
        if candidate_signal == "SHORT":
            return bias == "down"
        return False

    def _resolve_htf_bias(self, *, htf_candles: CandleSeries | None) -> tuple[Literal["up", "down", "flat"], dict[str, object]]:
        cfg = self.strategy_profile.filters.htf_bias
        info: dict[str, object] = {
            "enabled": bool(cfg.directional_filter),
            "timeframe": str(cfg.timeframe),
            "ma_type": str(cfg.ma_type),
            "period": int(cfg.period),
            "status": "disabled" if not cfg.directional_filter else "unavailable",
            "direction": "flat",
            "ma_value": None,
            "close_value": None,
        }

        if not cfg.directional_filter:
            return "flat", info
        if htf_candles is None:
            return "flat", info

        try:
            htf_candles.validate()
        except Exception as exc:
            info["status"] = "invalid"
            info["error"] = f"{exc.__class__.__name__}: {exc}"
            return "flat", info

        close = ensure_float64_contiguous(htf_candles.closes)
        if close.size < 3:
            info["status"] = "insufficient_data"
            return "flat", info

        period = max(1, int(cfg.period))
        if cfg.ma_type == "EMA":
            ma_arr = ema(close, period)
        else:
            ma_arr = self._hma(close, period)

        idx = close.size - 2 if close.size >= 2 else close.size - 1
        ma_value = float(ma_arr[idx]) if not np.isnan(ma_arr[idx]) else float("nan")
        close_value = float(close[idx])
        info["close_value"] = round(close_value, 8)

        if np.isnan(ma_value):
            info["status"] = "insufficient_data"
            return "flat", info

        info["ma_value"] = round(ma_value, 8)
        epsilon = max(abs(ma_value) * 1e-8, 1e-10)
        if close_value > ma_value + epsilon:
            direction: Literal["up", "down", "flat"] = "up"
        elif close_value < ma_value - epsilon:
            direction = "down"
        else:
            direction = "flat"

        info["status"] = "ok"
        info["direction"] = direction
        info[f"{cfg.ma_type.lower()}_{cfg.timeframe}_bias"] = direction
        return direction, info

    @staticmethod
    def _hour_in_window(*, hour: int, start: int, end: int) -> bool:
        if start == end:
            return True
        if start < end:
            return start <= hour < end
        return hour >= start or hour < end

    @staticmethod
    def _is_choppy_market(
        *,
        row: dict[str, object],
        btc_context: MarketContext | None,
        cfg,
    ) -> bool:
        adx = float(row.get("ADX", 0.0) or 0.0)
        atr_pct = float(row.get("ATR_PCT", 0.0) or 0.0)
        local_chop = adx < float(cfg.chop_adx_threshold) or atr_pct < float(cfg.btc_vol_threshold)
        btc_chop = bool(btc_context.meta.get("is_chop_market", False)) if isinstance(getattr(btc_context, "meta", None), dict) else False
        return bool(local_chop or btc_chop)

    def _resolve_candidate_signal(
        self,
        *,
        row: dict[str, object],
        candles: CandleSeries,
        allow_trend_continuation: bool,
        trigger_mode: Literal["WT", "WT_HMA_COMBO", "HMA_CROSS"],
    ) -> tuple[str | None, dict[str, object]]:
        base_signal = get_flip_candidate_signal(row)
        trigger_info: dict[str, object] = {
            "trigger_mode": trigger_mode,
            "trigger_pass": True,
            "blocked_reason": None,
            "blocked_stage": None,
            "source": "wt_flip",
            "synthetic_flip": False,
        }

        if base_signal is None and allow_trend_continuation:
            trend = int(row.get("TREND", 0))
            if trend > 0:
                base_signal = "LONG"
            elif trend < 0:
                base_signal = "SHORT"
            if base_signal is not None:
                trigger_info["source"] = "trend_continuation"
                trigger_info["synthetic_flip"] = True

        hma_snapshot = self._resolve_ltf_hma_snapshot(candles)
        trigger_info["hma_snapshot"] = hma_snapshot

        if trigger_mode == "WT":
            if base_signal is None:
                trigger_info["trigger_pass"] = False
                trigger_info["blocked_reason"] = "NO_FLIP"
                trigger_info["blocked_stage"] = "candidate_trigger"
            return base_signal, trigger_info

        if trigger_mode == "WT_HMA_COMBO":
            if base_signal is None:
                trigger_info["trigger_pass"] = False
                trigger_info["blocked_reason"] = "WT_TRIGGER_MISSING"
                trigger_info["blocked_stage"] = "candidate_trigger"
                return None, trigger_info
            if not self._is_hma_aligned(signal=base_signal, hma_direction=str(hma_snapshot.get("direction", "flat"))):
                trigger_info["trigger_pass"] = False
                trigger_info["blocked_reason"] = "LTF_HMA_MISMATCH"
                trigger_info["blocked_stage"] = "candidate_trigger"
                return None, trigger_info
            return base_signal, trigger_info

        # HMA_CROSS mode
        cross_signal_raw = hma_snapshot.get("cross_signal")
        cross_signal = str(cross_signal_raw) if isinstance(cross_signal_raw, str) else None
        if cross_signal in {"LONG", "SHORT"}:
            trigger_info["source"] = "hma_cross"
            trigger_info["synthetic_flip"] = True
            return cross_signal, trigger_info
        trigger_info["trigger_pass"] = False
        trigger_info["blocked_reason"] = "HMA_CROSS_MISSING"
        trigger_info["blocked_stage"] = "candidate_trigger"
        return None, trigger_info

    def _resolve_ltf_hma_snapshot(self, candles: CandleSeries) -> dict[str, object]:
        out: dict[str, object] = {
            "period": 20,
            "status": "unavailable",
            "direction": "flat",
            "cross_signal": None,
            "close_value": None,
            "hma_value": None,
        }
        try:
            candles.validate()
        except Exception as exc:
            out["status"] = "invalid"
            out["error"] = f"{exc.__class__.__name__}: {exc}"
            return out

        close = ensure_float64_contiguous(candles.closes)
        if close.size < 22:
            out["status"] = "insufficient_data"
            return out

        period = int(out["period"])
        hma_arr = self._hma(close, period)
        idx = close.size - 2
        prev_idx = idx - 1
        if idx < 1:
            out["status"] = "insufficient_data"
            return out
        hma_value = float(hma_arr[idx]) if not np.isnan(hma_arr[idx]) else float("nan")
        prev_hma = float(hma_arr[prev_idx]) if not np.isnan(hma_arr[prev_idx]) else float("nan")
        if np.isnan(hma_value) or np.isnan(prev_hma):
            out["status"] = "insufficient_data"
            return out
        close_value = float(close[idx])
        prev_close = float(close[prev_idx])
        out["close_value"] = round(close_value, 8)
        out["hma_value"] = round(hma_value, 8)

        epsilon = max(abs(hma_value) * 1e-8, 1e-10)
        rel_now = close_value - hma_value
        rel_prev = prev_close - prev_hma

        if rel_now > epsilon:
            out["direction"] = "up"
        elif rel_now < -epsilon:
            out["direction"] = "down"
        else:
            out["direction"] = "flat"

        cross_signal: str | None = None
        if rel_prev <= epsilon and rel_now > epsilon:
            cross_signal = "LONG"
        elif rel_prev >= -epsilon and rel_now < -epsilon:
            cross_signal = "SHORT"
        out["cross_signal"] = cross_signal
        out["status"] = "ok"
        return out

    @staticmethod
    def _is_hma_aligned(*, signal: str, hma_direction: str) -> bool:
        if signal == "LONG":
            return hma_direction == "up"
        if signal == "SHORT":
            return hma_direction == "down"
        return False

    @staticmethod
    def _wma(values: np.ndarray, period: int) -> np.ndarray:
        out = np.full(values.shape, np.nan, dtype=np.float64)
        if period <= 0 or values.size < period:
            return out
        weights = np.arange(1, period + 1, dtype=np.float64)
        denom = float(weights.sum())
        for idx in range(period - 1, values.size):
            window = values[idx - period + 1 : idx + 1]
            if np.isnan(window).any():
                continue
            out[idx] = float(np.dot(window, weights) / denom)
        return out

    def _hma(self, values: np.ndarray, period: int) -> np.ndarray:
        p = max(1, int(period))
        half = max(1, p // 2)
        sqrt_p = max(1, int(math.sqrt(p)))
        wma_half = self._wma(values, half)
        wma_full = self._wma(values, p)
        diff = (2.0 * wma_half) - wma_full
        return self._wma(diff, sqrt_p)
