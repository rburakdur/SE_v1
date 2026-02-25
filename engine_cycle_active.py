def process_active_real_position(sym: str, df, btc_ctx: dict) -> bool:
    """
    Aktif gercek pozisyonu gunceller.
    Donus: True ise coin dongusunda bu sembol icin devam etme (continue), False ise normal akisa devam.
    """
    import app as A

    if sym not in A.state.active_positions:
        return False

    pos = A.state.active_positions[sym]
    curr_h = float(df["high"].iloc[-1])
    curr_l = float(df["low"].iloc[-1])
    curr_c = float(df["close"].iloc[-1])
    entry_p = pos.get("entry_p", curr_c)

    pos["curr_p"] = curr_c
    closed = False
    close_reason = ""
    pnl = 0.0

    pos_time = A.datetime.strptime(pos["full_time"], "%Y-%m-%d %H:%M:%S")
    hold_minutes = (A.get_tr_time() - pos_time).total_seconds() / 60

    curr_pnl_live = ((curr_c - entry_p) / entry_p * 100 if pos["dir"] == "LONG" else (entry_p - curr_c) / entry_p * 100)
    pos["curr_pnl"] = curr_pnl_live
    pos["best_pnl"] = max(float(pos.get("best_pnl", curr_pnl_live)), curr_pnl_live)

    if pos["dir"] == "LONG":
        if curr_h >= pos["tp"]:
            pnl = (pos["tp"] - entry_p) / entry_p * 100
            closed, close_reason = True, "KAR ALDI"
        elif curr_l <= pos["sl"]:
            pnl = (pos["sl"] - entry_p) / entry_p * 100
            closed, close_reason = True, "STOP OLDU"
    else:
        if curr_l <= pos["tp"]:
            pnl = (entry_p - pos["tp"]) / entry_p * 100
            closed, close_reason = True, "KAR ALDI"
        elif curr_h >= pos["sl"]:
            pnl = (entry_p - pos["sl"]) / entry_p * 100
            closed, close_reason = True, "STOP OLDU"

    if (not closed) and hold_minutes > A.CONFIG["MAX_HOLD_MINUTES"]:
        last_trend = int(df["TREND"].iloc[-1])
        grace_limit = A.CONFIG["MAX_HOLD_MINUTES"] + A.CONFIG["MAX_HOLD_ST_GRACE_BARS"] * 5
        adverse_flip = ((pos["dir"] == "LONG" and last_trend == -1) or (pos["dir"] == "SHORT" and last_trend == 1))

        if curr_pnl_live > 0:
            if pos["dir"] == "LONG":
                pos["sl"] = max(float(pos["sl"]), float(entry_p))
            else:
                pos["sl"] = min(float(pos["sl"]), float(entry_p))

        if adverse_flip:
            closed = True
            close_reason = "TREND_FLIP_EXIT"
            pnl = curr_pnl_live
        elif hold_minutes > grace_limit:
            best_pnl = float(pos.get("best_pnl", curr_pnl_live))
            ema20_now = float(df["EMA20"].iloc[-1])
            ema_against = ((pos["dir"] == "LONG" and curr_c < ema20_now) or (pos["dir"] == "SHORT" and curr_c > ema20_now))
            no_progress = (
                curr_pnl_live <= float(A.CONFIG["STALE_EXIT_MIN_PNL_PCT"])
                and best_pnl < float(A.CONFIG["STALE_EXIT_MIN_BEST_PNL_PCT"])
            )
            if no_progress or ema_against:
                closed = True
                close_reason = "STALE_EXIT"
                pnl = curr_pnl_live

    if not closed:
        return False

    A.state.cooldowns[sym] = A.get_tr_time()
    trade_size = pos.get("trade_size", A.CONFIG["MIN_TRADE_SIZE"])
    pnl_usd = trade_size * (pnl / 100)
    A.state.update_balance(pnl, trade_size)

    row_at_close = df.iloc[-2]
    trade_log = {
        "Trade_Mode": "REAL",
        "Scan_ID": A.state.scan_id,
        "Restarted": pos.get("restarted", False),
        "Tarih": A.get_trading_day_str(),
        "Giris_Saati": pos["full_time"].split(" ")[1],
        "Cikis_Saati": A.get_tr_time().strftime("%H:%M:%S"),
        "Hold_Dakika": round(hold_minutes, 1),
        "Coin": sym,
        "Yon": pos["dir"],
        "Giris_Fiyati": round(entry_p, 6),
        "Cikis_Fiyati": round(curr_c, 6),
        "TP_Seviyesi": round(pos["tp"], 6),
        "SL_Seviyesi": round(pos["sl"], 6),
        "TP_SL_Orani": round(abs(pos["tp"] - entry_p) / abs(pos["sl"] - entry_p), 3),
        "Risk_USD": round(trade_size, 2),
        "PnL_Yuzde": round(pnl, 2),
        "PnL_USD": round(pnl_usd, 2),
        "Kasa_Son_Durum": round(A.state.balance, 2),
        "Sonuc": close_reason,
        "Giris_RSI": pos.get("entry_rsi", 0),
        "Giris_ADX": pos.get("entry_adx", 0),
        "Giris_VOL_RATIO": pos.get("entry_vol_ratio", 0),
        "Giris_ATR_PCT": pos.get("entry_atr_pct", 0),
        "Giris_Power_Score": pos.get("power_score", 0),
        "Giris_Score": pos.get("signal_score", 0),
        "Cikis_RSI": round(float(row_at_close["RSI"]), 2),
        "Cikis_ADX": round(float(row_at_close["ADX"]), 2),
        "Cikis_VOL_RATIO": round(float(row_at_close["VOL_RATIO"]), 3),
        "Cikis_ATR_PCT": round(float(row_at_close["ATR_PCT"]), 4),
        "Cikis_TREND": int(row_at_close["TREND"]),
        "Cikis_MACD_HIST": round(float(row_at_close.get("MACD_HIST", 0)), 6),
        "BTC_Trend": btc_ctx.get("trend", 0),
        "BTC_ATR_PCT": btc_ctx.get("atr_pct", 0.0),
        "BTC_RSI": btc_ctx.get("rsi", 0.0),
        "BTC_ADX": btc_ctx.get("adx", 0.0),
        "BTC_Vol_Ratio": btc_ctx.get("vol_ratio", 0.0),
        "Cfg_ST_M": A.CONFIG["ST_M"],
        "Cfg_RSI_Long": A.CONFIG["RSI_LONG"],
        "Cfg_RSI_Short": A.CONFIG["RSI_SHORT"],
        "Cfg_VOL_Filter": A.CONFIG["VOL_FILTER"],
        "Cfg_ADX_Thr": A.CONFIG["ADX_THRESHOLD"],
        "Cfg_SL_M": A.CONFIG["SL_M"],
        "Cfg_TP_M": A.CONFIG["TP_M"],
    }
    A.log_trade_to_csv(trade_log)

    try:
        chart_buf = A.create_trade_chart(df, sym, pos, is_entry=False, curr_c=curr_c, pnl=pnl, close_reason=close_reason)
        tag_emoji = "green_circle,moneybag" if pnl > 0 else "red_circle,x"
        A.send_ntfy_notification(
            f"GERCEK ISLEM KAPANDI: {sym}",
            f"Sonuc: {close_reason}\nPnL: %{pnl:.2f} | Kâr/Zarar: ${pnl_usd:.2f}\n"
            f"Sure: {round(hold_minutes,1)} dk | Yeni Kasa: ${A.state.balance:.2f}",
            image_buf=chart_buf,
            tags=tag_emoji,
            priority="4",
        )
    except Exception as e:
        A.log_error("close_notification", e, sym)

    del A.state.active_positions[sym]
    A.state.save_state()
    return True

