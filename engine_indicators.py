import numpy as np
import pandas as pd
import talib


def _talib_ready_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64, copy=False)
    return np.ascontiguousarray(arr, dtype=np.float64)


def hesapla_indikatorler(df: pd.DataFrame, config: dict, symbol: str = "") -> pd.DataFrame:
    c = _talib_ready_array(df["close"])
    h = _talib_ready_array(df["high"])
    l = _talib_ready_array(df["low"])
    v = _talib_ready_array(df["volume"])
    if min(len(c), len(h), len(l), len(v)) < 50:
        raise ValueError("insufficient bars after numeric conversion")
    if (
        (not np.isfinite(c[-50:]).all())
        or (not np.isfinite(h[-50:]).all())
        or (not np.isfinite(l[-50:]).all())
        or (not np.isfinite(v[-50:]).all())
    ):
        raise ValueError("non-finite values in recent OHLCV")

    out = df.copy()
    out["RSI"] = talib.RSI(c, int(config["RSI_PERIOD"]))
    out["ADX"] = talib.ADX(h, l, c, 14)
    out["PLUS_DI"] = talib.PLUS_DI(h, l, c, 14)
    out["MINUS_DI"] = talib.MINUS_DI(h, l, c, 14)
    out["EMA20"] = talib.EMA(c, 20)
    out["EMA50"] = talib.EMA(c, 50)
    out["ATR_14"] = talib.ATR(h, l, c, 14)
    out["BBANDS_UP"], out["BBANDS_MID"], out["BBANDS_LOW"] = talib.BBANDS(c, 20, 2, 2)
    out["MACD"], out["MACD_SIGNAL"], out["MACD_HIST"] = talib.MACD(c, 12, 26, 9)
    out["VOL_SMA_20"] = talib.SMA(v, 20)
    out["VOL_RATIO"] = np.where(out["VOL_SMA_20"] > 0, v / out["VOL_SMA_20"], 0)
    out["ATR_PCT"] = (out["ATR_14"] / out["close"]) * 100

    atr_st = talib.ATR(h, l, c, 10)
    hl2 = (h + l) / 2
    st_line = np.zeros(len(c))
    trend = np.ones(len(c))
    st_m = float(config["ST_M"])

    for i in range(1, len(c)):
        if np.isnan(atr_st[i]) or np.isnan(hl2[i]):
            st_line[i] = st_line[i - 1]
            trend[i] = trend[i - 1]
            continue
        up = hl2[i] + st_m * atr_st[i]
        dn = hl2[i] - st_m * atr_st[i]
        if c[i - 1] > st_line[i - 1]:
            st_line[i] = max(dn, st_line[i - 1])
            trend[i] = 1
        else:
            st_line[i] = min(up, st_line[i - 1])
            trend[i] = -1
        if trend[i] != trend[i - 1]:
            st_line[i] = dn if trend[i] == 1 else up

    out["TREND"] = trend
    out["ST_LINE"] = st_line
    out["ST_DIST_PCT"] = ((out["close"] - out["ST_LINE"]) / out["close"]) * 100
    out["FLIP_LONG"] = (out["TREND"] == 1) & (out["TREND"].shift(1) == -1)
    out["FLIP_SHORT"] = (out["TREND"] == -1) & (out["TREND"].shift(1) == 1)

    out["BODY_PCT"] = abs(out["close"] - out["open"]) / out["open"] * 100
    out["UPPER_WICK"] = (out["high"] - out[["close", "open"]].max(axis=1)) / out["open"] * 100
    out["LOWER_WICK"] = (out[["close", "open"]].min(axis=1) - out["low"]) / out["open"] * 100
    return out

