import pandas as pd


def safe_api_get(url: str, params=None, retries=5):
    import app as A

    for attempt in range(retries):
        try:
            r = A.requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                A.time.sleep(10)
            else:
                A.time.sleep(2)
        except A.requests.exceptions.RequestException as e:
            A.log_error("safe_api_get", e, f"url={url} attempt={attempt}")
            A.time.sleep(3)
    return None


def get_top_futures_coins(limit=30) -> list:
    data = safe_api_get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if data:
        def is_ascii_clean(s):
            try:
                s.encode("ascii")
                return True
            except UnicodeEncodeError:
                return False

        usdt_pairs = [
            d for d in data
            if d["symbol"].endswith("USDT")
            and "_" not in d["symbol"]
            and is_ascii_clean(d["symbol"])
        ]
        return [
            p["symbol"]
            for p in sorted(usdt_pairs, key=lambda x: float(x["quoteVolume"]), reverse=True)[:limit]
        ]
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def get_live_futures_data(symbol: str, limit=300):
    data = safe_api_get(
        "https://fapi.binance.com/fapi/v1/klines",
        {"symbol": symbol, "interval": "5m", "limit": limit},
    )
    if data:
        df = pd.DataFrame(data).iloc[:, :6]
        df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms") + pd.Timedelta(hours=3)
        df.set_index("timestamp", inplace=True)
        return df.apply(pd.to_numeric, errors="coerce")
    return None


def get_btc_context() -> dict:
    import app as A

    btc_df = get_live_futures_data("BTCUSDT", 200)
    if btc_df is None or len(btc_df) < 50:
        return {
            "trend": 0, "atr_pct": 0.0, "rsi": 0.0, "adx": 0.0,
            "vol_ratio": 0.0, "macd_hist": 0.0, "close": 0.0,
            "ema20": 0.0, "bb_width_pct": 0.0
        }
    try:
        btc_df = A.hesapla_indikatorler(btc_df, "BTCUSDT")
        r = btc_df.iloc[-2]
        bb_width = (r["BBANDS_UP"] - r["BBANDS_LOW"]) / r["BBANDS_MID"] * 100 if r["BBANDS_MID"] > 0 else 0
        return {
            "trend": int(r["TREND"]),
            "atr_pct": round(float(r["ATR_PCT"]), 4),
            "rsi": round(float(r["RSI"]), 2),
            "adx": round(float(r["ADX"]), 2),
            "vol_ratio": round(float(r["VOL_RATIO"]), 3),
            "macd_hist": round(float(r["MACD_HIST"]), 6),
            "close": round(float(r["close"]), 2),
            "ema20": round(float(r["EMA20"]), 2),
            "bb_width_pct": round(float(bb_width), 3),
        }
    except Exception as e:
        try:
            dtypes_str = ",".join(
                f"{c}:{btc_df[c].dtype}" for c in ["open", "high", "low", "close", "volume"] if c in btc_df.columns
            )
        except Exception:
            dtypes_str = "dtype_unavailable"
        A.log_error("get_btc_context", e, dtypes_str)
        return {
            "trend": 0, "atr_pct": 0.0, "rsi": 0.0, "adx": 0.0,
            "vol_ratio": 0.0, "macd_hist": 0.0, "close": 0.0,
            "ema20": 0.0, "bb_width_pct": 0.0
        }

