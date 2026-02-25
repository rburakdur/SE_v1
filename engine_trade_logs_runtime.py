import os

import pandas as pd


def log_trade_to_csv(trade_dict: dict):
    import app as A

    try:
        pd.DataFrame([trade_dict]).to_csv(
            A.FILES["LOG"],
            mode="a",
            header=not os.path.exists(A.FILES["LOG"]),
            index=False,
            encoding="utf-8-sig",
        )
    except Exception as e:
        A.log_error("log_trade_to_csv", e)


def log_potential_signal(
    sym: str,
    signal_type: str,
    row,
    score: int,
    power_score: float,
    entered: bool,
    reason: str = "",
    btc_ctx: dict = None,
    candidate_signal: str = None,
    candidate_flags: dict = None,
    auto_flags: dict = None,
    decision_meta: dict = None,
):
    import app as A

    if btc_ctx is None:
        btc_ctx = {}
    candidate_flags = candidate_flags or {}
    auto_flags = auto_flags or {}
    decision_meta = decision_meta or {}

    try:
        bb_width = (float(row["BBANDS_UP"]) - float(row["BBANDS_LOW"])) / max(float(row["BBANDS_MID"]), 1e-10) * 100
        log_row = {
            "timestamp": A.get_tr_time().isoformat(),
            "scan_id": A.state.scan_id,
            "coin": sym,
            "signal": signal_type,
            "is_candidate": bool(candidate_signal),
            "candidate_signal": candidate_signal or "",
            "score": score,
            "power_score": power_score,
            "close": round(float(row["close"]), 6),
            "open": round(float(row["open"]), 6),
            "high": round(float(row["high"]), 6),
            "low": round(float(row["low"]), 6),
            "volume": round(float(row["volume"]), 2),
            "rsi": round(float(row["RSI"]), 2),
            "adx": round(float(row["ADX"]), 2),
            "plus_di": round(float(row["PLUS_DI"]), 2),
            "minus_di": round(float(row["MINUS_DI"]), 2),
            "atr_14": round(float(row["ATR_14"]), 6),
            "atr_pct": round(float(row["ATR_PCT"]), 4),
            "vol_ratio": round(float(row["VOL_RATIO"]), 3),
            "ema20": round(float(row["EMA20"]), 6),
            "ema50": round(float(row["EMA50"]), 6),
            "st_line": round(float(row["ST_LINE"]), 6),
            "st_dist_pct": round(float(row["ST_DIST_PCT"]), 4),
            "macd": round(float(row["MACD"]), 6),
            "macd_signal": round(float(row["MACD_SIGNAL"]), 6),
            "macd_hist": round(float(row["MACD_HIST"]), 6),
            "bb_upper": round(float(row["BBANDS_UP"]), 6),
            "bb_lower": round(float(row["BBANDS_LOW"]), 6),
            "bb_width_pct": round(float(bb_width), 3),
            "body_pct": round(float(row["BODY_PCT"]), 4),
            "upper_wick_pct": round(float(row["UPPER_WICK"]), 4),
            "lower_wick_pct": round(float(row["LOWER_WICK"]), 4),
            "btc_trend": btc_ctx.get("trend", 0),
            "btc_atr_pct": btc_ctx.get("atr_pct", 0.0),
            "btc_rsi": btc_ctx.get("rsi", 0.0),
            "btc_adx": btc_ctx.get("adx", 0.0),
            "btc_vol_ratio": btc_ctx.get("vol_ratio", 0.0),
            "btc_macd_hist": btc_ctx.get("macd_hist", 0.0),
            "btc_close": btc_ctx.get("close", 0.0),
            "btc_bb_width_pct": btc_ctx.get("bb_width_pct", 0.0),
            "entered": bool(entered),
            "rejection_reason": reason if not entered else "",
            "candidate_flip_ok": bool(candidate_flags.get("flip_ok", False)),
            "candidate_rsi_ok": bool(candidate_flags.get("rsi_ok", False)),
            "candidate_vol_ok": bool(candidate_flags.get("vol_ok", False)),
            "candidate_adx_ok": bool(candidate_flags.get("adx_ok", False)),
            "candidate_atr_ok": bool(candidate_flags.get("atr_ok", False)),
            "candidate_ema_ok": bool(candidate_flags.get("ema_ok", False)),
            "candidate_all_ok": bool(candidate_flags.get("all_ok", False)),
            "auto_flip_ok": bool(auto_flags.get("flip_ok", False)),
            "auto_rsi_ok": bool(auto_flags.get("rsi_ok", False)),
            "auto_vol_ok": bool(auto_flags.get("vol_ok", False)),
            "auto_adx_ok": bool(auto_flags.get("adx_ok", False)),
            "auto_atr_ok": bool(auto_flags.get("atr_ok", False)),
            "auto_ema_ok": bool(auto_flags.get("ema_ok", False)),
            "auto_all_ok": bool(auto_flags.get("all_ok", False)),
            "mkt_is_chop": bool(A.state.is_chop_market),
            "mkt_dir_text": str(A.state.market_direction_text),
            "decision_power_ok": bool(decision_meta.get("power_ok", False)),
            "decision_btc_trend_ok": bool(decision_meta.get("btc_trend_ok", False)),
            "decision_chop_ok": bool(decision_meta.get("chop_ok", False)),
            "decision_cooldown_ok": bool(decision_meta.get("cooldown_ok", False)),
            "decision_capacity_ok": bool(decision_meta.get("capacity_ok", False)),
            "decision_already_in": bool(decision_meta.get("already_in", False)),
            "decision_auto_entry_eligible": bool(decision_meta.get("auto_entry_eligible", False)),
            "rejection_stage": str(decision_meta.get("rejection_stage", "")),
        }
        pd.DataFrame([log_row]).to_csv(
            A.FILES["ALL_SIGNALS"],
            mode="a",
            header=not os.path.exists(A.FILES["ALL_SIGNALS"]),
            index=False,
        )
    except Exception as e:
        A.log_error("log_potential_signal", e, sym)


def log_market_context(btc_ctx: dict, coin_count: int, open_pos: int):
    import app as A

    try:
        row = {
            "timestamp": A.get_tr_time().isoformat(),
            "scan_id": A.state.scan_id,
            "btc_trend": btc_ctx.get("trend", 0),
            "btc_atr_pct": btc_ctx.get("atr_pct", 0.0),
            "btc_rsi": btc_ctx.get("rsi", 0.0),
            "btc_adx": btc_ctx.get("adx", 0.0),
            "btc_vol_ratio": btc_ctx.get("vol_ratio", 0.0),
            "btc_macd_hist": btc_ctx.get("macd_hist", 0.0),
            "btc_close": btc_ctx.get("close", 0.0),
            "btc_bb_width_pct": btc_ctx.get("bb_width_pct", 0.0),
            "is_chop": A.state.is_chop_market,
            "market_dir": A.state.market_direction_text,
            "coins_scanned": coin_count,
            "open_positions": open_pos,
            "balance": round(A.state.balance, 2),
        }
        pd.DataFrame([row]).to_csv(
            A.FILES["MARKET_CONTEXT"],
            mode="a",
            header=not os.path.exists(A.FILES["MARKET_CONTEXT"]),
            index=False,
        )
    except Exception as e:
        A.log_error("log_market_context", e)

