def finalize_cycle_and_sleep(fetched_data):
    import app as A

    try:
        del fetched_data
    except Exception:
        pass
    A.gc.collect()

    A.state.is_scanning = False
    A.rotate_logs()

    current_time = A.get_tr_time()
    current_hour = current_time.hour
    A.state.maybe_reset_daily_balance(current_time)

    if current_hour != A.state.last_heartbeat_hour:
        A.state.last_heartbeat_hour = current_hour

        real_stats = A.get_trade_performance_snapshot("REAL")
        virtual_stats = A.get_trade_performance_snapshot("VIRTUAL")
        tot_trd = real_stats["total_trades"]
        wins = real_stats["wins"]
        b_wr = real_stats["win_rate"]
        pf = real_stats["pf"]
        max_dd = real_stats["max_dd"]
        v_tot = virtual_stats["total_trades"]
        v_wins = virtual_stats["wins"]
        v_wr = virtual_stats["win_rate"]
        v_pf = virtual_stats["pf"]
        r_pnl_usd = real_stats["net_pnl_usd"]
        v_pnl_usd = virtual_stats["net_pnl_usd"]

        hb_title, hb_msg = A.format_hourly_report_message(
            current_time=current_time,
            real_metrics=(tot_trd, wins, b_wr, pf, max_dd),
            virtual_metrics=(v_tot, v_wins, v_wr, v_pf, 0.0),
            r_pnl_usd=r_pnl_usd,
            v_pnl_usd=v_pnl_usd,
        )
        A.send_ntfy_notification(hb_title, hb_msg, tags="clipboard,bar_chart", priority="3")
        A.state.hourly_missed_signals = 0

    trading_day = A.get_trading_day_str(current_time)
    if current_hour == 2 and current_time.minute >= 56 and A.state.last_dump_trading_day != trading_day:
        A.state.last_dump_trading_day = trading_day
        A.state.save_state()
        A.gunluk_dump_gonder()

    now = A.get_tr_time()
    target = now.replace(second=0, microsecond=0)
    next_m = next((m for m in A.CONFIG["TARGET_MINUTES"] if m > now.minute), A.CONFIG["TARGET_MINUTES"][0])
    if next_m == A.CONFIG["TARGET_MINUTES"][0]:
        target += A.timedelta(hours=1)
    target = target.replace(minute=next_m)

    A.state.status = (
        f"ğŸ’¤ SENKRON BEKLEME (Sonraki Tarama: "
        f"{target.strftime('%H:%M:%S')} | "
        f"Bu Scan Kacirilan: {A.state.missed_this_scan})"
    )
    A.draw_fund_dashboard()

    sleep_sec = (target - now).total_seconds()
    if sleep_sec > 0:
        A.time.sleep(sleep_sec)

