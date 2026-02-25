def create_trade_chart(df, sym, pos, is_entry=False, curr_c=None, pnl=0.0, close_reason=""):
    import app as A

    try:
        A.plt.style.use("dark_background")
        fig, ax = A.plt.subplots(figsize=(10, 5))
        entry_dt = A.pd.to_datetime(pos.get("entry_idx_time", pos.get("full_time", A.get_tr_time().strftime("%Y-%m-%d %H:%M:%S"))))

        plot_df = (
            df.tail(60).copy()
            if is_entry
            else df[df.index >= entry_dt - A.pd.Timedelta(minutes=150)].tail(200).copy()
        )

        up = plot_df[plot_df.close >= plot_df.open]
        down = plot_df[plot_df.close < plot_df.open]

        ax.vlines(up.index, up.low, up.high, color="#2ecc71", linewidth=1.5, alpha=0.8)
        ax.vlines(down.index, down.low, down.high, color="#e74c3c", linewidth=1.5, alpha=0.8)
        bar_w = (plot_df.index[-1] - plot_df.index[0]).total_seconds() / len(plot_df) * 0.6 / 86400
        ax.bar(up.index, up.close - up.open, bar_w, bottom=up.open, color="#2ecc71", alpha=0.9)
        ax.bar(down.index, down.open - down.close, bar_w, bottom=down.close, color="#e74c3c", alpha=0.9)

        ax.axhline(pos["entry_p"], color="#3498db", linestyle="--", alpha=0.8, label="Giris")
        ax.axhline(pos["tp"], color="#2ecc71", linestyle=":", linewidth=2, label="TP")
        ax.axhline(pos["sl"], color="#e74c3c", linestyle=":", linewidth=2, label="SL")

        if is_entry:
            ax.scatter(entry_dt, pos["entry_p"], color="yellow", s=150, zorder=5, edgecolors="black")
        else:
            ax.scatter(entry_dt, pos["entry_p"], color="yellow", s=120, zorder=5, edgecolors="black")
            exit_price = pos["tp"] if "KAR" in close_reason else pos["sl"]
            ax.scatter(
                plot_df.index[-1],
                exit_price,
                color="#2ecc71" if pnl > 0 else "#e74c3c",
                s=200,
                zorder=5,
                marker="X",
                edgecolors="white",
            )

        ax.xaxis.set_major_formatter(A.mdates.DateFormatter("%H:%M"))
        fig.autofmt_xdate(rotation=30)
        ax.legend(loc="upper left", framealpha=0.3)

        buf = A.io.BytesIO()
        A.plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        A.plt.close("all")
        buf.seek(0)
        return buf
    except Exception as e:
        A.log_error("create_trade_chart", e, sym)
        A.plt.close("all")
        return None


def maybe_open_paper_position(sym: str, df, row_closed, signal: str, power_score: float, signal_score: int, reason: str = ""):
    import app as A

    try:
        if not signal:
            return
        if sym in A.state.paper_positions:
            return
        entry_p = float(row_closed["close"])
        atr_val = float(row_closed["ATR_14"])
        sl_p = entry_p - (A.CONFIG["SL_M"] * atr_val) if signal == "LONG" else entry_p + (A.CONFIG["SL_M"] * atr_val)
        tp_p = entry_p + (A.CONFIG["TP_M"] * atr_val) if signal == "LONG" else entry_p - (A.CONFIG["TP_M"] * atr_val)
        v_size = A.state.dynamic_trade_size

        A.state.paper_positions[sym] = {
            "dir": signal,
            "entry_p": entry_p,
            "sl": sl_p,
            "tp": tp_p,
            "full_time": A.get_tr_time().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_idx_time": str(row_closed.name),
            "curr_pnl": 0.0,
            "best_pnl": 0.0,
            "curr_p": float(df["close"].iloc[-1]),
            "trade_size": v_size,
            "entry_rsi": round(float(row_closed["RSI"]), 2),
            "entry_adx": round(float(row_closed["ADX"]), 2),
            "entry_vol_ratio": round(float(row_closed["VOL_RATIO"]), 3),
            "entry_atr_pct": round(float(row_closed["ATR_PCT"]), 4),
            "power_score": float(power_score),
            "signal_score": int(signal_score),
            "candidate_reason": reason or "SHORTLIST",
        }
        A.state.save_state()

        try:
            chart_buf = create_trade_chart(df, sym, A.state.paper_positions[sym], is_entry=True)
            tp_pct, sl_pct, rr = A.calc_tp_sl_metrics(entry_p, sl_p, tp_p, signal)
            A.send_ntfy_notification(
                f"SANAL ISLEM ACILDI: {sym}",
                f"Yon: {signal} | Fiyat: {entry_p:.5f}\n"
                f"SL: {sl_p:.5f} (-%{sl_pct:.2f}) | TP: {tp_p:.5f} (+%{tp_pct:.2f})\n"
                f"Hedef/Risk (R:R): {rr:.2f}\n"
                f"Pozisyon(sim): ${v_size:.2f} | Power: {power_score:.1f} | Score: {signal_score}/6\n"
                f"Neden: {reason or 'SHORTLIST'}",
                image_buf=chart_buf,
                tags="mag,chart_with_upwards_trend",
                priority="3",
            )
        except Exception as e:
            A.log_error("paper_entry_notification", e, sym)
    except Exception as e:
        A.log_error("maybe_open_paper_position", e, sym)


def update_paper_position_for_symbol(sym: str, df, btc_ctx: dict):
    import app as A

    if sym not in A.state.paper_positions:
        return
    try:
        pos = A.state.paper_positions[sym]
        curr_h = float(df["high"].iloc[-1])
        curr_l = float(df["low"].iloc[-1])
        curr_c = float(df["close"].iloc[-1])
        entry_p = float(pos.get("entry_p", curr_c))

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
                closed, close_reason, pnl = True, "TREND_FLIP_EXIT", curr_pnl_live
            elif hold_minutes > grace_limit:
                best_pnl = float(pos.get("best_pnl", curr_pnl_live))
                ema20_now = float(df["EMA20"].iloc[-1])
                ema_against = ((pos["dir"] == "LONG" and curr_c < ema20_now) or (pos["dir"] == "SHORT" and curr_c > ema20_now))
                no_progress = (
                    curr_pnl_live <= float(A.CONFIG["STALE_EXIT_MIN_PNL_PCT"])
                    and best_pnl < float(A.CONFIG["STALE_EXIT_MIN_BEST_PNL_PCT"])
                )
                if no_progress or ema_against:
                    closed, close_reason, pnl = True, "STALE_EXIT", curr_pnl_live

        if not closed:
            return

        trade_size = float(pos.get("trade_size", A.state.dynamic_trade_size))
        pnl_usd = trade_size * (pnl / 100)
        row_at_close = df.iloc[-2]
        trade_log = {
            "Trade_Mode": "VIRTUAL",
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
            "TP_SL_Orani": round(abs(pos["tp"] - entry_p) / max(abs(pos["sl"] - entry_p), 1e-12), 3),
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
        }
        A.log_trade_to_csv(trade_log)

        try:
            chart_buf = create_trade_chart(df, sym, pos, is_entry=False, curr_c=curr_c, pnl=pnl, close_reason=close_reason)
            A.send_ntfy_notification(
                f"SANAL ISLEM KAPANDI: {sym}",
                f"Sonuc: {close_reason}\nPnL: %{pnl:.2f} | Sim PnL: ${pnl_usd:.2f}\n"
                f"Sure: {round(hold_minutes,1)} dk | Yon: {pos['dir']}\n"
                f"Giris: {entry_p:.5f} | Cikis: {curr_c:.5f}",
                image_buf=chart_buf,
                tags="clipboard,chart_with_upwards_trend",
                priority="3",
            )
        except Exception as e:
            A.log_error("paper_close_notification", e, sym)

        del A.state.paper_positions[sym]
        A.state.save_state()
    except Exception as e:
        A.log_error("update_paper_position_for_symbol", e, sym)

