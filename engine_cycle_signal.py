def process_signal_for_coin(sym: str, df, btc_ctx: dict):
    import app as A

    row_closed = df.iloc[-2]

    candidate_signal = A.get_flip_candidate_signal(row_closed)
    signal = None
    entered, reason = False, ""

    if not candidate_signal:
        return

    cand_t = A.get_signal_thresholds("candidate")
    auto_t = A.get_signal_thresholds("auto")
    candidate_flags = A.evaluate_signal_filters(row_closed, candidate_signal, cand_t)
    auto_flags = A.evaluate_signal_filters(row_closed, candidate_signal, auto_t)
    power_score = float(A.hesapla_power_score(row_closed, cand_t))
    auto_power_score = float(A.hesapla_power_score(row_closed, auto_t))
    signal_score = int(A.hesapla_signal_score(row_closed, candidate_signal, cand_t))
    auto_signal_score = int(A.hesapla_signal_score(row_closed, candidate_signal, auto_t))
    log_signal_type = candidate_signal
    shortlist_eligible = False
    decision_meta = {
        "power_ok": power_score >= cand_t["min_power_score"],
        "btc_trend_ok": True,
        "chop_ok": True,
        "cooldown_ok": True,
        "capacity_ok": True,
        "already_in": False,
        "auto_entry_eligible": False,
        "rejection_stage": "",
        "rejection_reason": "",
    }

    if not candidate_flags["all_ok"]:
        reason = A.get_candidate_fail_reason(row_closed, candidate_signal)
        decision_meta["rejection_stage"] = "candidate_filter"
        decision_meta["rejection_reason"] = reason
    elif power_score < cand_t["min_power_score"]:
        reason = f"CAND_LOW_POWER_{power_score:.0f}"
        decision_meta["power_ok"] = False
        decision_meta["rejection_stage"] = "candidate_filter"
        decision_meta["rejection_reason"] = reason
    else:
        shortlist_eligible = True
        signal = candidate_signal if auto_flags["all_ok"] else None
        log_signal_type = signal if signal else candidate_signal
        if signal is None:
            reason = "AUTO_TECH_FAIL"
            decision_meta["rejection_stage"] = "auto_filter"
            decision_meta["rejection_reason"] = reason
        else:
            effective_auto_power = auto_power_score
            btc_trend_match = False
            if btc_ctx["trend"] != 0:
                btc_trend_match = ((signal == "LONG" and btc_ctx["trend"] == 1) or (signal == "SHORT" and btc_ctx["trend"] == -1))
            decision_meta["btc_trend_ok"] = bool(btc_trend_match)

            if A.CONFIG.get("AUTO_BTC_TREND_MODE", "hard_block") == "soft_penalty":
                if not btc_trend_match:
                    effective_auto_power -= float(A.CONFIG.get("AUTO_BTC_TREND_PENALTY", 0.0))
            else:
                if btc_ctx["trend"] == 0:
                    reason = "BTC_VERI_YOK"
                elif not btc_trend_match:
                    reason = "BTC_TREND_KOTU"
                if reason:
                    decision_meta["rejection_stage"] = "market_filter"
                    decision_meta["rejection_reason"] = reason

            if not reason and A.state.is_chop_market:
                chop_policy = str(A.CONFIG.get("AUTO_CHOP_POLICY", "block")).lower()
                if chop_policy == "block":
                    decision_meta["chop_ok"] = False
                    reason = "CHOP_MARKET"
                    decision_meta["rejection_stage"] = "market_filter"
                    decision_meta["rejection_reason"] = reason
                elif chop_policy == "penalty":
                    effective_auto_power -= float(A.CONFIG.get("AUTO_CHOP_PENALTY", 0.0))

            if not reason and effective_auto_power < auto_t["min_power_score"]:
                reason = f"LOW_POWER_{effective_auto_power:.0f}"
                decision_meta["power_ok"] = False
                decision_meta["rejection_stage"] = "auto_filter"
                decision_meta["rejection_reason"] = reason

            if not reason and sym in A.state.active_positions:
                reason = "ALREADY_IN"
                decision_meta["already_in"] = True
                decision_meta["rejection_stage"] = "execution_filter"
                decision_meta["rejection_reason"] = reason

            if (
                not reason
                and sym in A.state.cooldowns
                and (A.get_tr_time() - A.state.cooldowns[sym]).total_seconds() / 60 < A.CONFIG["COOLDOWN_MINUTES"]
            ):
                cd_left = A.CONFIG["COOLDOWN_MINUTES"] - (A.get_tr_time() - A.state.cooldowns[sym]).total_seconds() / 60
                reason = f"COOLDOWN_{round(cd_left,1)}dk"
                decision_meta["cooldown_ok"] = False
                decision_meta["rejection_stage"] = "execution_filter"
                decision_meta["rejection_reason"] = reason

            if not reason and len(A.state.active_positions) >= A.CONFIG["MAX_POSITIONS"]:
                reason = "MAX_POS"
                decision_meta["capacity_ok"] = False
                decision_meta["rejection_stage"] = "execution_filter"
                decision_meta["rejection_reason"] = reason

            if not reason and float(row_closed["ADX"]) < float(A.CONFIG["CHOP_ADX_THRESHOLD"]):
                reason = "LOW_ADX"
                decision_meta["rejection_stage"] = "execution_filter"
                decision_meta["rejection_reason"] = reason

            if not reason:
                decision_meta["auto_entry_eligible"] = True
                power_score = effective_auto_power
                signal_score = auto_signal_score

    if decision_meta.get("auto_entry_eligible", False):
        entry_p = float(row_closed["close"])
        atr_val = float(row_closed["ATR_14"])
        sl_p = entry_p - (A.CONFIG["SL_M"] * atr_val) if signal == "LONG" else entry_p + (A.CONFIG["SL_M"] * atr_val)
        tp_p = entry_p + (A.CONFIG["TP_M"] * atr_val) if signal == "LONG" else entry_p - (A.CONFIG["TP_M"] * atr_val)
        t_size = A.state.dynamic_trade_size

        A.state.active_positions[sym] = {
            "dir": signal,
            "entry_p": entry_p,
            "sl": sl_p,
            "tp": tp_p,
            "full_time": A.get_tr_time().strftime("%Y-%m-%d %H:%M:%S"),
            "entry_idx_time": str(row_closed.name),
            "curr_pnl": 0.0,
            "best_pnl": 0.0,
            "curr_p": float(df["close"].iloc[-1]),
            "trade_size": t_size,
            "entry_rsi": round(float(row_closed["RSI"]), 2),
            "entry_adx": round(float(row_closed["ADX"]), 2),
            "entry_vol_ratio": round(float(row_closed["VOL_RATIO"]), 3),
            "entry_atr_pct": round(float(row_closed["ATR_PCT"]), 4),
            "power_score": power_score,
            "signal_score": signal_score,
        }
        A.state.save_state()
        entered = True

        try:
            chart_buf = A.create_trade_chart(df, sym, A.state.active_positions[sym], is_entry=True)
            tp_pct, sl_pct, rr = A.calc_tp_sl_metrics(entry_p, sl_p, tp_p, signal)
            A.send_ntfy_notification(
                f"GERCEK ISLEM ACILDI: {sym}",
                f"Yon: {signal} | Fiyat: {entry_p:.5f}\n"
                f"SL: {sl_p:.5f} (-%{sl_pct:.2f}) | TP: {tp_p:.5f} (+%{tp_pct:.2f})\n"
                f"Hedef/Risk (R:R): {rr:.2f}\n"
                f"Pozisyon: ${t_size:.2f} | Power: {power_score} | Score: {signal_score}/6",
                image_buf=chart_buf,
                tags="chart_with_upwards_trend",
                priority="4",
            )
        except Exception as e:
            A.log_error("entry_notification", e, sym)

    if shortlist_eligible:
        A.maybe_open_paper_position(
            sym=sym,
            df=df,
            row_closed=row_closed,
            signal=candidate_signal,
            power_score=float(A.hesapla_power_score(row_closed, cand_t)),
            signal_score=int(A.hesapla_signal_score(row_closed, candidate_signal, cand_t)),
            reason=reason if reason else "SHORTLIST_OK",
        )

    if signal and not entered:
        A.state.missed_this_scan += 1
        A.state.hourly_missed_signals += 1

    if signal:
        A.state.son_sinyaller.append(
            {
                "zaman": A.get_tr_time().strftime("%H:%M:%S"),
                "coin": sym,
                "signal": signal,
                "entered": entered,
                "reason": reason if not entered else "-",
                "power": power_score,
                "rsi": round(float(row_closed["RSI"]), 1),
                "adx": round(float(row_closed["ADX"]), 1),
                "vol": round(float(row_closed["VOL_RATIO"]), 2),
            }
        )
        if len(A.state.son_sinyaller) > 50:
            A.state.son_sinyaller = A.state.son_sinyaller[-50:]

    A.log_potential_signal(
        sym,
        log_signal_type,
        row_closed,
        signal_score,
        power_score,
        entered,
        reason,
        btc_ctx,
        candidate_signal=candidate_signal,
        candidate_flags=candidate_flags,
        auto_flags=auto_flags,
        decision_meta=decision_meta,
    )

