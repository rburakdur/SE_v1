def run_main_loop():
    import app as A

    A.atexit.register(lambda: A.flush_state_on_shutdown("atexit"))
    try:
        A.signal.signal(A.signal.SIGTERM, A._handle_termination)
        A.signal.signal(A.signal.SIGINT, A._handle_termination)
    except Exception as e:
        A.log_error("signal_handler_setup", e)
    A.log_storage_diagnostics()

    start_msg = (
        f"ğŸ’µ Guncel Kasa: ${A.state.balance:.2f}\n"
        f"ğŸ›¡ï¸ Global Crash Guard Aktif\n"
        f"ğŸ“Š Maksimum Loglama Modu: hunter_history + all_signals + market_context + error_log\n"
        f"ğŸ”¢ Walk-Forward Hazir: Power Score + Signal Score + Config Snapshot\n"
        f"ğŸ’¾ Log Rotation & RAM Korumasi Devrede\n"
        f"Scan ID: {A.state.scan_id} | v87.0 production modunda baÅŸlatildi!"
    )
    A.log_print("=" * 50)
    A.log_print("RBD-CRYPT v87.0 BASLATILDI")
    A.log_print(f"Kasa: ${A.state.balance:.2f} | Scan ID: {A.state.scan_id}")
    A.log_print("=" * 50)
    A.send_ntfy_notification("ğŸš€ v87.0 BASLATILDI", start_msg, tags="rocket,shield", priority="4")

    komut_thread = A.threading.Thread(target=A.ntfy_komut_dinle, daemon=True)
    komut_thread.start()

    if A.state.active_positions:
        pozlar = ", ".join(f"{sym} {pos['dir']} @ {pos['entry_p']:.5f}" for sym, pos in A.state.active_positions.items())
        A.send_ntfy_notification(
            "ğŸ”„ RESTART â€” Pozisyonlar Kurtarildi",
            f"{len(A.state.active_positions)} acik pozisyon devam ediyor:\n{pozlar}\nSure sayaci resetlendi.",
            tags="arrows_counterclockwise,white_check_mark",
            priority="4",
        )
    if A.state.paper_positions:
        spoz = ", ".join(f"{sym} {pos['dir']} @ {pos['entry_p']:.5f}" for sym, pos in A.state.paper_positions.items())
        A.send_ntfy_notification(
            "RESTART - SANAL POZISYONLAR",
            f"{len(A.state.paper_positions)} acik sanal pozisyon devam ediyor:\n{spoz}\nSure sayaci resetlendi.",
            tags="arrows_counterclockwise,mag",
            priority="3",
        )

    while True:
        try:
            A.run_bot_cycle()
        except Exception as e:
            error_msg = (
                f"Sistem Hata Aldi ve coktu!\n"
                f"Hata: {str(e)[:150]}\n"
                f"Scan ID: {A.state.scan_id}\n"
                f"30 Saniye icinde kendini onarip tekrar baÅŸlayacak."
            )
            A.log_error("MAIN_LOOP", e)
            A.log_print(f"CRITICAL ERROR: {e}")
            A.send_ntfy_notification(
                "ğŸš¨ SISTEM coKTu (RESTART ATILIYOR)",
                error_msg,
                tags="rotating_light,warning",
                priority="5",
            )
            A.time.sleep(30)

