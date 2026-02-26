from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np

from ..config import LegacyParitySettings
from ..indicators.talib_indicators import adx, atr, ema, ensure_float64_contiguous, rsi, sma
from ..models.market_context import MarketContext
from ..models.symbol_state import SymbolBarState
from .signal_engine import CandleSeries


@dataclass(slots=True)
class LegacyThresholds:
    rsi_long: float
    rsi_short: float
    vol_filter: float
    adx_threshold: float
    min_atr_pct: float
    min_power_score: float


@dataclass(slots=True)
class LegacyAnalysis:
    closed_row: dict[str, float | bool | int | datetime]
    live_row: dict[str, float | bool | int | datetime]
    symbol_state: SymbolBarState


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if values.size < period:
        return out
    for i in range(period - 1, values.size):
        window = values[i - period + 1 : i + 1]
        if np.isfinite(window).all():
            out[i] = np.std(window, ddof=0)
    return out


def _bbands(close: np.ndarray, period: int = 20, n_dev: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = sma(close, period)
    std = _rolling_std(close, period)
    up = mid + (std * n_dev)
    low = mid - (std * n_dev)
    return up, mid, low


def _macd_hist(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    macd = ema(close, 12) - ema(close, 26)
    signal = ema(macd, 9)
    hist = macd - signal
    return macd, signal, hist


def _supertrend_like(close: np.ndarray, high: np.ndarray, low: np.ndarray, cfg: LegacyParitySettings) -> tuple[np.ndarray, np.ndarray]:
    atr_st = atr(high, low, close, period=cfg.supertrend_atr_period)
    hl2 = (high + low) / 2.0
    st_line = np.zeros(close.size, dtype=np.float64)
    trend = np.ones(close.size, dtype=np.int64)
    st_m = float(cfg.supertrend_multiplier)
    for i in range(1, close.size):
        if np.isnan(atr_st[i]) or np.isnan(hl2[i]):
            st_line[i] = st_line[i - 1]
            trend[i] = trend[i - 1]
            continue
        up = hl2[i] + st_m * atr_st[i]
        dn = hl2[i] - st_m * atr_st[i]
        if close[i - 1] > st_line[i - 1]:
            st_line[i] = max(dn, st_line[i - 1])
            trend[i] = 1
        else:
            st_line[i] = min(up, st_line[i - 1])
            trend[i] = -1
        if trend[i] != trend[i - 1]:
            st_line[i] = dn if trend[i] == 1 else up
    return trend, st_line


def _float(x: float | np.floating) -> float:
    if np.isnan(x):
        return 0.0
    return float(x)


def _build_row(
    *,
    idx: int,
    candles: CandleSeries,
    rsi_arr: np.ndarray,
    adx_arr: np.ndarray,
    ema20_arr: np.ndarray,
    ema50_arr: np.ndarray,
    atr14_arr: np.ndarray,
    atr_pct_arr: np.ndarray,
    bb_up: np.ndarray,
    bb_mid: np.ndarray,
    bb_low: np.ndarray,
    macd_hist_arr: np.ndarray,
    vol_ratio_arr: np.ndarray,
    trend_arr: np.ndarray,
    flip_long_arr: np.ndarray,
    flip_short_arr: np.ndarray,
) -> dict[str, float | bool | int | datetime]:
    return {
        "timestamp": candles.open_times[idx],
        "open": float(candles.opens[idx]),
        "high": float(candles.highs[idx]),
        "low": float(candles.lows[idx]),
        "close": float(candles.closes[idx]),
        "volume": float(candles.volumes[idx]),
        "RSI": _float(rsi_arr[idx]),
        "ADX": _float(adx_arr[idx]),
        "EMA20": _float(ema20_arr[idx]),
        "EMA50": _float(ema50_arr[idx]),
        "ATR_14": _float(atr14_arr[idx]),
        "ATR_PCT": _float(atr_pct_arr[idx]),
        "BBANDS_UP": _float(bb_up[idx]),
        "BBANDS_MID": _float(bb_mid[idx]),
        "BBANDS_LOW": _float(bb_low[idx]),
        "MACD_HIST": _float(macd_hist_arr[idx]),
        "VOL_RATIO": _float(vol_ratio_arr[idx]),
        "TREND": int(trend_arr[idx]),
        "FLIP_LONG": bool(flip_long_arr[idx]),
        "FLIP_SHORT": bool(flip_short_arr[idx]),
    }


def analyze_candles(candles: CandleSeries, cfg: LegacyParitySettings) -> LegacyAnalysis:
    candles.validate()
    close = ensure_float64_contiguous(candles.closes)
    high = ensure_float64_contiguous(candles.highs)
    low = ensure_float64_contiguous(candles.lows)
    vol = ensure_float64_contiguous(candles.volumes)
    if close.size < 60:
        raise ValueError("Need at least 60 candles for legacy parity analysis")

    rsi_arr = rsi(close, period=cfg.rsi_period)
    adx_arr = adx(high, low, close, period=14)
    ema20_arr = ema(close, 20)
    ema50_arr = ema(close, 50)
    atr14_arr = atr(high, low, close, period=14)
    atr_pct_arr = np.divide(atr14_arr, close, out=np.full_like(close, np.nan), where=close != 0.0) * 100.0
    bb_up, bb_mid, bb_low = _bbands(close, period=20, n_dev=2.0)
    _, _, macd_hist_arr = _macd_hist(close)
    vol_sma20 = sma(vol, 20)
    vol_ratio_arr = np.divide(vol, vol_sma20, out=np.zeros_like(vol), where=vol_sma20 > 0.0)
    trend_arr, _ = _supertrend_like(close, high, low, cfg)
    prev_trend = np.roll(trend_arr, 1)
    prev_trend[0] = trend_arr[0]
    flip_long_arr = (trend_arr == 1) & (prev_trend == -1)
    flip_short_arr = (trend_arr == -1) & (prev_trend == 1)

    closed_idx = close.size - 2
    live_idx = close.size - 1
    closed_row = _build_row(
        idx=closed_idx,
        candles=candles,
        rsi_arr=rsi_arr,
        adx_arr=adx_arr,
        ema20_arr=ema20_arr,
        ema50_arr=ema50_arr,
        atr14_arr=atr14_arr,
        atr_pct_arr=atr_pct_arr,
        bb_up=bb_up,
        bb_mid=bb_mid,
        bb_low=bb_low,
        macd_hist_arr=macd_hist_arr,
        vol_ratio_arr=vol_ratio_arr,
        trend_arr=trend_arr,
        flip_long_arr=flip_long_arr,
        flip_short_arr=flip_short_arr,
    )
    live_row = _build_row(
        idx=live_idx,
        candles=candles,
        rsi_arr=rsi_arr,
        adx_arr=adx_arr,
        ema20_arr=ema20_arr,
        ema50_arr=ema50_arr,
        atr14_arr=atr14_arr,
        atr_pct_arr=atr_pct_arr,
        bb_up=bb_up,
        bb_mid=bb_mid,
        bb_low=bb_low,
        macd_hist_arr=macd_hist_arr,
        vol_ratio_arr=vol_ratio_arr,
        trend_arr=trend_arr,
        flip_long_arr=flip_long_arr,
        flip_short_arr=flip_short_arr,
    )
    symbol_state = SymbolBarState(
        symbol="",
        current_bar_time=candles.open_times[live_idx],
        current_high=float(candles.highs[live_idx]),
        current_low=float(candles.lows[live_idx]),
        current_close=float(candles.closes[live_idx]),
        current_trend=int(live_row["TREND"]),
        current_ema20=float(live_row["EMA20"]),
        current_adx=float(live_row["ADX"]),
        current_rsi=float(live_row["RSI"]),
        current_vol_ratio=float(live_row["VOL_RATIO"]),
        current_atr_pct=float(live_row["ATR_PCT"]),
        closed_bar_time=candles.open_times[closed_idx],
        closed_close=float(candles.closes[closed_idx]),
        closed_atr14=float(closed_row["ATR_14"]),
        closed_trend=int(closed_row["TREND"]),
        closed_ema20=float(closed_row["EMA20"]),
        closed_rsi=float(closed_row["RSI"]),
        closed_adx=float(closed_row["ADX"]),
        closed_vol_ratio=float(closed_row["VOL_RATIO"]),
        closed_atr_pct=float(closed_row["ATR_PCT"]),
        closed_macd_hist=float(closed_row["MACD_HIST"]),
    )
    return LegacyAnalysis(closed_row=closed_row, live_row=live_row, symbol_state=symbol_state)


def get_signal_thresholds(cfg: LegacyParitySettings, layer: Literal["auto", "candidate"] = "auto") -> LegacyThresholds:
    if layer == "candidate":
        return LegacyThresholds(
            rsi_long=cfg.candidate_rsi_long,
            rsi_short=cfg.candidate_rsi_short,
            vol_filter=cfg.candidate_vol_filter,
            adx_threshold=cfg.candidate_adx_threshold,
            min_atr_pct=cfg.candidate_min_atr_percent,
            min_power_score=cfg.candidate_min_power_score,
        )
    return LegacyThresholds(
        rsi_long=cfg.auto_entry_rsi_long,
        rsi_short=cfg.auto_entry_rsi_short,
        vol_filter=cfg.auto_entry_vol_filter,
        adx_threshold=cfg.auto_entry_adx_threshold,
        min_atr_pct=cfg.auto_entry_min_atr_percent,
        min_power_score=cfg.auto_entry_min_power_score,
    )


def get_flip_candidate_signal(row: dict[str, object]) -> Literal["LONG", "SHORT"] | None:
    flip_long = bool(row["FLIP_LONG"])
    flip_short = bool(row["FLIP_SHORT"])
    if flip_long and not flip_short:
        return "LONG"
    if flip_short and not flip_long:
        return "SHORT"
    if flip_long and flip_short:
        return "LONG" if int(row.get("TREND", 0)) >= 0 else "SHORT"
    return None


def evaluate_signal_filters(
    row: dict[str, object],
    signal_type: str | None,
    thresholds: LegacyThresholds,
) -> dict[str, bool]:
    if signal_type not in ("LONG", "SHORT"):
        return {"flip_ok": False, "rsi_ok": False, "vol_ok": False, "adx_ok": False, "atr_ok": False, "ema_ok": False, "all_ok": False}
    is_long = signal_type == "LONG"
    flip_ok = bool(row["FLIP_LONG"]) if is_long else bool(row["FLIP_SHORT"])
    rsi_ok = float(row["RSI"]) > thresholds.rsi_long if is_long else float(row["RSI"]) < thresholds.rsi_short
    vol_ok = float(row["VOL_RATIO"]) > thresholds.vol_filter
    adx_ok = float(row["ADX"]) > thresholds.adx_threshold
    atr_ok = float(row["ATR_PCT"]) >= thresholds.min_atr_pct
    ema_ok = float(row["close"]) > float(row["EMA20"]) if is_long else float(row["close"]) < float(row["EMA20"])
    all_ok = all([flip_ok, rsi_ok, vol_ok, adx_ok, atr_ok, ema_ok])
    return {
        "flip_ok": bool(flip_ok),
        "rsi_ok": bool(rsi_ok),
        "vol_ok": bool(vol_ok),
        "adx_ok": bool(adx_ok),
        "atr_ok": bool(atr_ok),
        "ema_ok": bool(ema_ok),
        "all_ok": bool(all_ok),
    }


def score_from_flags(flags: dict[str, bool]) -> int:
    return int(sum(1 for k in ["flip_ok", "rsi_ok", "vol_ok", "adx_ok", "atr_ok", "ema_ok"] if flags.get(k)))


def calculate_power_score(row: dict[str, object], thresholds: LegacyThresholds) -> tuple[float, dict[str, float]]:
    score = 0.0
    breakdown: dict[str, float] = {}
    if bool(row["FLIP_LONG"]):
        rsi_component = max(0.0, min(25.0, (float(row["RSI"]) - thresholds.rsi_long) * 2.5))
    else:
        rsi_component = max(0.0, min(25.0, (thresholds.rsi_short - float(row["RSI"])) * 2.5))
    score += rsi_component
    breakdown["rsi_component"] = round(rsi_component, 4)

    vol_component = max(0.0, min(25.0, (float(row["VOL_RATIO"]) - thresholds.vol_filter) * 15.0))
    score += vol_component
    breakdown["vol_component"] = round(vol_component, 4)

    adx_component = max(0.0, min(20.0, (float(row["ADX"]) - thresholds.adx_threshold) * 0.8))
    score += adx_component
    breakdown["adx_component"] = round(adx_component, 4)

    atr_component = max(0.0, min(15.0, (float(row["ATR_PCT"]) - thresholds.min_atr_pct) * 5.0))
    score += atr_component
    breakdown["atr_component"] = round(atr_component, 4)

    macd_component = 0.0
    if bool(row["FLIP_LONG"]) and float(row["MACD_HIST"]) > 0:
        macd_component = 10.0
    elif bool(row["FLIP_SHORT"]) and float(row["MACD_HIST"]) < 0:
        macd_component = 10.0
    score += macd_component
    breakdown["macd_component"] = round(macd_component, 4)

    bb_mid = max(float(row["BBANDS_MID"]), 1e-10)
    bb_width = (float(row["BBANDS_UP"]) - float(row["BBANDS_LOW"])) / bb_mid * 100.0
    bb_component = max(0.0, min(5.0, bb_width * 0.5))
    score += bb_component
    breakdown["bb_width_component"] = round(bb_component, 4)
    breakdown["bb_width_pct"] = round(bb_width, 4)
    return round(score, 2), breakdown


def get_candidate_fail_reason(row: dict[str, object], candidate_signal: str, cfg: LegacyParitySettings) -> str:
    t = get_signal_thresholds(cfg, "candidate")
    is_long = candidate_signal == "LONG"
    if is_long and not (float(row["RSI"]) > t.rsi_long):
        return "CAND_FAIL_RSI"
    if (not is_long) and not (float(row["RSI"]) < t.rsi_short):
        return "CAND_FAIL_RSI"
    if not (float(row["VOL_RATIO"]) > t.vol_filter):
        return "CAND_FAIL_VOL"
    if is_long and not (float(row["close"]) > float(row["EMA20"])):
        return "CAND_FAIL_EMA"
    if (not is_long) and not (float(row["close"]) < float(row["EMA20"])):
        return "CAND_FAIL_EMA"
    if not (float(row["ADX"]) > t.adx_threshold):
        return "CAND_FAIL_ADX"
    if not (float(row["ATR_PCT"]) >= t.min_atr_pct):
        return "CAND_FAIL_ATR"
    return "CAND_FAIL_OTHER"


def _trend_code_to_direction(trend: int) -> Literal["up", "down", "flat"]:
    if trend > 0:
        return "up"
    if trend < 0:
        return "down"
    return "flat"


def derive_btc_context_from_analysis(
    analysis: LegacyAnalysis,
    *,
    interval: str,
    symbol: str,
    cfg: LegacyParitySettings,
) -> MarketContext:
    r = analysis.closed_row
    trend_code = int(r["TREND"])
    atr_pct = float(r["ATR_PCT"])
    chop_state: Literal["trending", "choppy", "unknown"] = "choppy" if atr_pct < cfg.btc_vol_threshold else "trending"
    return MarketContext(
        symbol=symbol,
        interval=interval,
        bar_time=r["timestamp"],  # type: ignore[arg-type]
        trend_direction=_trend_code_to_direction(trend_code),
        trend_score=float(abs(trend_code) * 100),
        chop_state=chop_state,
        metrics={
            "atr_pct": atr_pct,
            "rsi": float(r["RSI"]),
            "adx": float(r["ADX"]),
            "vol_ratio": float(r["VOL_RATIO"]),
            "macd_hist": float(r["MACD_HIST"]),
            "close": float(r["close"]),
            "ema20": float(r["EMA20"]),
            "bb_width_pct": (
                ((float(r["BBANDS_UP"]) - float(r["BBANDS_LOW"])) / max(float(r["BBANDS_MID"]), 1e-10)) * 100.0
            ),
            "trend_code": float(trend_code),
        },
        meta={
            "trend_code": trend_code,
            "is_chop_market": atr_pct < cfg.btc_vol_threshold,
        },
    )


def btc_trend_match(signal_type: str, btc_context: MarketContext | None) -> bool:
    if btc_context is None:
        return False
    trend_code = int(btc_context.meta.get("trend_code", 0)) if isinstance(btc_context.meta, dict) else 0
    if trend_code == 0:
        return False
    return (signal_type == "LONG" and trend_code == 1) or (signal_type == "SHORT" and trend_code == -1)
