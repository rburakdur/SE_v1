_shutdown_flag = {"done": False}


def log_print(msg: str):
    import app as A

    zaman = A.get_tr_time().strftime("%H:%M:%S")
    print(f"[{zaman}] {msg}", flush=True)


def log_storage_diagnostics():
    import app as A

    try:
        log_print(f"STORAGE PATH: {A.CONFIG['BASE_PATH']}")
        for key in ["ACTIVE", "STATE"]:
            p = A.FILES[key]
            exists = A.os.path.exists(p)
            size = A.os.path.getsize(p) if exists else 0
            log_print(f"  {key}: {'OK' if exists else 'MISSING'} | {p} | {size} bytes")
        log_print(f"  Recovered active positions: {len(getattr(A.state, 'active_positions', {}))}")
        log_print(f"  Recovered paper positions: {len(getattr(A.state, 'paper_positions', {}))}")
        log_print(
            f"  GitHub backup: {'ENABLED' if A.CONFIG.get('GITHUB_BACKUP_ENABLED') else 'DISABLED'} | "
            f"Repo: {A.CONFIG.get('GITHUB_BACKUP_REPO','-')} | Branch: {A.CONFIG.get('GITHUB_BACKUP_BRANCH','main')}"
        )
    except Exception as e:
        A.log_error("log_storage_diagnostics", e)


def flush_state_on_shutdown(reason: str = "shutdown"):
    import app as A

    if _shutdown_flag["done"]:
        return
    _shutdown_flag["done"] = True
    try:
        if A.CONFIG.get("ENABLE_SHUTDOWN_FLUSH", True):
            A.state.save_state()
            log_print(f"STATE FLUSH OK ({reason})")
    except Exception as e:
        A.log_error("flush_state_on_shutdown", e, reason)


def handle_termination(signum, frame):
    import app as A

    try:
        log_print(f"TERM SIGNAL ALINDI: {signum}")
        flush_state_on_shutdown(f"signal_{signum}")
        try:
            A.send_ntfy_notification(
                "BOT SHUTDOWN",
                f"Signal: {signum}\nScan ID: {A.state.scan_id}\nAcik pozisyon: {len(A.state.active_positions)}",
                tags="warning",
                priority="3",
            )
        except Exception as e:
            A.log_error("shutdown_ntfy", e, str(signum))
    finally:
        raise SystemExit(0)


def draw_fund_dashboard():
    import app as A

    tot_trd, wins, b_wr, pf, max_dd = A.get_advanced_metrics()
    kasa_ok = "+" if A.state.balance >= A.CONFIG["STARTING_BALANCE"] else "-"

    print("-" * 70, flush=True)
    log_print(f"RBD-CRYPT v87.0 | Scan #{A.state.scan_id}")
    log_print(
        f"PIYASA : {A.state.market_direction_text} | BTC ATR%: {A.state.btc_atr_pct:.3f} | "
        f"BTC RSI: {A.state.btc_rsi:.1f} | BTC ADX: {A.state.btc_adx:.1f}"
    )
    log_print(
        f"KASA   : ${A.state.balance:.2f} ({kasa_ok}) | Tepe: ${A.state.peak_balance:.2f} | "
        f"Pozisyon/islem: ${A.state.dynamic_trade_size:.1f}"
    )
    log_print(f"PERFORMANS: {wins}/{tot_trd} islem (%{b_wr} basari) | PF: {pf} | Max DD: %{max_dd:.2f}")

    if A.state.active_positions:
        log_print(f"ACIK POZISYONLAR ({len(A.state.active_positions)}/{A.CONFIG['MAX_POSITIONS']}):")
        for sym, pos in A.state.active_positions.items():
            curr_pnl = pos.get("curr_pnl", 0.0)
            dur = str(A.get_tr_time() - A.datetime.strptime(pos["full_time"], "%Y-%m-%d %H:%M:%S")).split(".")[0]
            yon = "LONG" if pos.get("dir") == "LONG" else "SHORT"
            isaret = "+" if curr_pnl >= 0 else ""
            log_print(
                f"  >> {sym} {yon} | Sure: {dur} | PnL: {isaret}{curr_pnl:.2f}% | "
                f"Giris: {pos.get('entry_p',0):.5f} | TP: {pos.get('tp',0):.5f} | SL: {pos.get('sl',0):.5f}"
            )
    else:
        log_print(f"ACIK POZISYON: Yok (0/{A.CONFIG['MAX_POSITIONS']})")

    if A.state.son_sinyaller:
        log_print(f"SON SINYALLER (son {min(8, len(A.state.son_sinyaller))}):")
        for s in A.state.son_sinyaller[-8:]:
            durum = "GIRILDI" if s["entered"] else f"REDDEDILDI({s['reason']})"
            log_print(
                f"  {s['zaman']} {s['coin']:15s} {s['signal']:5s} | {durum:30s} | "
                f"Power:{s['power']:5.0f} RSI:{s['rsi']:5.1f} ADX:{s['adx']:5.1f} VOL:{s['vol']:.2f}x"
            )
    else:
        log_print("SON SINYALLER: Henuz sinyal tespit edilmedi.")

    if A.state.is_scanning:
        total = A.state.total_count if A.state.total_count > 0 else 1
        done = A.state.processed_count
        pct = int((done / total) * 100)
        filled = int(20 * done / total)
        bar = "#" * filled + "." * (20 - filled)
        log_print(f"TARAMA : [{bar}] %{pct} ({done}/{total}) | Simdi: {A.state.current_coin}")
    else:
        log_print(f"DURUM  : {A.state.status}")
    print("-" * 70, flush=True)


def _reset_generated_runtime_data():
    import app as A

    base = A.CONFIG["BASE_PATH"]
    deleted = []
    failed = []

    primary_keys = ["LOG", "ALL_SIGNALS", "MARKET_CONTEXT", "ERROR_LOG", "ACTIVE", "PAPER_ACTIVE", "STATE"]
    for key in primary_keys:
        p = A.FILES.get(key)
        if not p:
            continue
        try:
            if A.os.path.exists(p):
                A.os.remove(p)
                deleted.append(p)
            tmp_p = p + ".tmp"
            if A.os.path.exists(tmp_p):
                A.os.remove(tmp_p)
                deleted.append(tmp_p)
        except Exception as e:
            failed.append((p, type(e).__name__))
            A.log_error("reset_generated_remove_primary", e, p)

    prefixes = (
        "hunter_history.csv_old_",
        "hunter_history.csv_old_schema_",
        "all_signals.csv_old_",
        "all_signals.csv_old_schema_",
        "market_context.csv_old_",
        "market_context.csv_old_schema_",
        "error_log.csv_old_",
        "error_log.csv_old_schema_",
        "daily_backup_",
    )
    try:
        for root, _, files in A.os.walk(base):
            for fname in files:
                if not any(fname.startswith(px) for px in prefixes):
                    continue
                fp = A.os.path.join(root, fname)
                try:
                    A.os.remove(fp)
                    deleted.append(fp)
                except Exception as e:
                    failed.append((fp, type(e).__name__))
                    A.log_error("reset_generated_remove_rotated", e, fp)
    except Exception as e:
        A.log_error("reset_generated_walk", e, base)

    try:
        A.state.active_positions = {}
        A.state.paper_positions = {}
        A.state.cooldowns = {}
        A.state.son_sinyaller = []
        A.state.processed_count = 0
        A.state.total_count = 0
        A.state.current_coin = "Reset"
        A.state.is_scanning = False
        A.state.status = "MANUAL RESET (NTFY)"
        A.state.missed_this_scan = 0
        A.state.hourly_missed_signals = 0
        A.state.balance = float(A.CONFIG["STARTING_BALANCE"])
        A.state.peak_balance = float(A.CONFIG["STARTING_BALANCE"])
        A.state.scan_id = 0
        A.state.last_dump_trading_day = ""
        A.state.last_balance_reset_trading_day = ""
        A.state.btc_atr_pct = 0.0
        A.state.btc_rsi = 0.0
        A.state.btc_adx = 0.0
        A.state.btc_vol_ratio = 0.0
        A.state.is_chop_market = False
        try:
            A.state.last_heartbeat_hour = A.get_tr_time().hour
        except Exception:
            pass
    except Exception:
        pass

    return deleted, failed


def ntfy_komut_dinle():
    import app as A

    url = f"https://ntfy.sh/{A.CONFIG['NTFY_TOPIC']}/sse"
    while True:
        try:
            with A.requests.get(url, stream=True, timeout=(10, 90)) as resp:
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8", errors="ignore")
                    if not line.startswith("data:"):
                        continue
                    try:
                        payload = A.json.loads(line[5:].strip())
                        baslik = payload.get("title", "")
                        if any(x in baslik for x in ["ISLEM", "BASLATILDI", "coKTu", "Rapor", "Dokum", "RESTART", "Durum"]):
                            continue
                        mesaj = payload.get("message", "").strip().lower()
                        if not mesaj:
                            continue

                        log_print(f"NTFY KOMUT ALINDI: {mesaj}")

                        if mesaj == "logs":
                            A.send_ntfy_notification(
                                "ğŸ“¦ Manuel Log Talebi Alindi",
                                "Dosyalar hazirlaniyor, birazdan gelecek...",
                                tags="package",
                                priority="3",
                            )
                            A.threading.Thread(target=A.gunluk_dump_gonder, daemon=True).start()
                        elif mesaj in ("resetlogs", "logreset", "logsil", "clearlogs", "resetall", "fullreset", "temizbasla"):
                            deleted, failed = _reset_generated_runtime_data()
                            A.send_ntfy_notification(
                                "FULL RESET OK",
                                (
                                    f"Silinen dosya (log/json/backup): {len(deleted)}\n"
                                    f"Hata: {len(failed)}\n"
                                    "Temizlenenler: loglar + active/paper/state json + backup zip\n"
                                    f"Acik pozisyon: Gercek {len(A.state.active_positions)} | Sanal {len(A.state.paper_positions)}\n"
                                    f"Kasa reset: ${float(A.state.balance):.2f} | Scan ID: {int(A.state.scan_id)}"
                                ),
                                tags="wastebasket,arrows_counterclockwise",
                                priority="4" if not failed else "3",
                            )
                        elif mesaj in ("durum", "status"):
                            real_stats = A.get_trade_performance_snapshot("REAL")
                            virtual_stats = A.get_trade_performance_snapshot("VIRTUAL")
                            durum_msg = A.engine_build_status_message(
                                balance=float(A.state.balance),
                                peak_balance=float(A.state.peak_balance),
                                max_positions=int(A.CONFIG["MAX_POSITIONS"]),
                                active_positions=A.state.active_positions,
                                real_stats=real_stats,
                                virtual_stats=virtual_stats,
                                market_direction_text=str(A.state.market_direction_text),
                                scan_id=int(A.state.scan_id),
                            )
                            A.send_ntfy_notification(
                                f"ğŸ“Š Anlik Durum ({A.get_tr_time().strftime('%H:%M')})",
                                durum_msg,
                                tags="bar_chart",
                                priority="3",
                            )
                    except Exception as e:
                        A.log_error("ntfy_komut_parse", e, line[:100])
        except Exception as e:
            A.log_error("ntfy_komut_dinle", e)
            A.time.sleep(15)
