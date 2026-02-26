from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def ensure_float64_contiguous(values: Iterable[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float64)
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


def ema(values: Iterable[float] | np.ndarray, period: int) -> np.ndarray:
    x = ensure_float64_contiguous(values)
    out = np.full_like(x, np.nan)
    if period <= 0 or x.size == 0:
        return out
    alpha = 2.0 / (period + 1.0)
    out[0] = x[0]
    for i in range(1, x.size):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def sma(values: Iterable[float] | np.ndarray, period: int) -> np.ndarray:
    x = ensure_float64_contiguous(values)
    out = np.full_like(x, np.nan)
    if period <= 0 or x.size < period:
        return out
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out[period - 1 :] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def rsi(close: Iterable[float] | np.ndarray, period: int = 14) -> np.ndarray:
    c = ensure_float64_contiguous(close)
    out = np.full_like(c, np.nan)
    if c.size < period + 1:
        return out
    delta = np.diff(c)
    gain = np.where(delta > 0.0, delta, 0.0)
    loss = np.where(delta < 0.0, -delta, 0.0)
    avg_gain = np.empty_like(delta)
    avg_loss = np.empty_like(delta)
    avg_gain[: period] = np.nan
    avg_loss[: period] = np.nan
    avg_gain[period - 1] = np.mean(gain[:period])
    avg_loss[period - 1] = np.mean(loss[:period])
    for i in range(period, delta.size):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.inf), where=avg_loss != 0.0)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    out[1:] = rsi_vals
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    return np.maximum(hl, np.maximum(hc, lc))


def wilder_smooth(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full_like(values, np.nan)
    if values.size < period:
        return out
    out[period - 1] = np.sum(values[:period])
    for i in range(period, values.size):
        out[i] = out[i - 1] - (out[i - 1] / period) + values[i]
    return out


def atr(
    high: Iterable[float] | np.ndarray,
    low: Iterable[float] | np.ndarray,
    close: Iterable[float] | np.ndarray,
    period: int = 14,
) -> np.ndarray:
    h = ensure_float64_contiguous(high)
    l = ensure_float64_contiguous(low)
    c = ensure_float64_contiguous(close)
    tr = true_range(h, l, c)
    smooth = wilder_smooth(tr, period)
    out = np.divide(smooth, period, out=np.full_like(smooth, np.nan), where=~np.isnan(smooth))
    return out


def adx(
    high: Iterable[float] | np.ndarray,
    low: Iterable[float] | np.ndarray,
    close: Iterable[float] | np.ndarray,
    period: int = 14,
) -> np.ndarray:
    h = ensure_float64_contiguous(high)
    l = ensure_float64_contiguous(low)
    c = ensure_float64_contiguous(close)
    out = np.full_like(c, np.nan)
    if c.size < (period * 2):
        return out
    up_move = h - np.roll(h, 1)
    down_move = np.roll(l, 1) - l
    up_move[0] = 0.0
    down_move[0] = 0.0
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)
    tr = true_range(h, l, c)
    tr_s = wilder_smooth(tr, period)
    plus_s = wilder_smooth(plus_dm, period)
    minus_s = wilder_smooth(minus_dm, period)
    plus_di = 100.0 * np.divide(plus_s, tr_s, out=np.zeros_like(plus_s), where=tr_s > 0.0)
    minus_di = 100.0 * np.divide(minus_s, tr_s, out=np.zeros_like(minus_s), where=tr_s > 0.0)
    dx = 100.0 * np.divide(
        np.abs(plus_di - minus_di),
        plus_di + minus_di,
        out=np.zeros_like(plus_di),
        where=(plus_di + minus_di) > 0.0,
    )
    first_adx_idx = (period * 2) - 2
    if first_adx_idx >= dx.size:
        return out
    adx_vals = np.full_like(dx, np.nan)
    adx_vals[first_adx_idx] = np.nanmean(dx[period - 1 : first_adx_idx + 1])
    for i in range(first_adx_idx + 1, dx.size):
        adx_vals[i] = ((adx_vals[i - 1] * (period - 1)) + dx[i]) / period
    out[:] = adx_vals
    return out


@dataclass(slots=True)
class IndicatorSnapshot:
    ema_fast: float
    ema_slow: float
    rsi: float
    atr: float
    atr_pct: float
    adx: float
    volume_ratio: float


def latest_snapshot(
    *,
    close: Iterable[float] | np.ndarray,
    high: Iterable[float] | np.ndarray,
    low: Iterable[float] | np.ndarray,
    volume: Iterable[float] | np.ndarray,
    ema_fast_period: int,
    ema_slow_period: int,
    rsi_period: int = 14,
    atr_period: int = 14,
    adx_period: int = 14,
    volume_period: int = 20,
) -> IndicatorSnapshot:
    c = ensure_float64_contiguous(close)
    h = ensure_float64_contiguous(high)
    l = ensure_float64_contiguous(low)
    v = ensure_float64_contiguous(volume)
    ema_fast_arr = ema(c, ema_fast_period)
    ema_slow_arr = ema(c, ema_slow_period)
    rsi_arr = rsi(c, rsi_period)
    atr_arr = atr(h, l, c, atr_period)
    adx_arr = adx(h, l, c, adx_period)
    vol_sma = sma(v, volume_period)
    vol_ratio_arr = np.divide(v, vol_sma, out=np.full_like(v, np.nan), where=vol_sma > 0.0)
    last_close = float(c[-1])
    last_atr = float(atr_arr[-1]) if not np.isnan(atr_arr[-1]) else 0.0
    atr_pct = (last_atr / last_close) if last_close else 0.0
    return IndicatorSnapshot(
        ema_fast=float(ema_fast_arr[-1]) if not np.isnan(ema_fast_arr[-1]) else 0.0,
        ema_slow=float(ema_slow_arr[-1]) if not np.isnan(ema_slow_arr[-1]) else 0.0,
        rsi=float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0,
        atr=last_atr,
        atr_pct=float(atr_pct),
        adx=float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 0.0,
        volume_ratio=float(vol_ratio_arr[-1]) if not np.isnan(vol_ratio_arr[-1]) else 1.0,
    )
