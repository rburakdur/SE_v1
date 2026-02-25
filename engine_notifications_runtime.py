def send_ntfy_notification(title: str, message: str, image_buf=None, tags="robot", priority="3"):
    import app as A

    return A.engine_send_ntfy_notification(
        topic=A.CONFIG["NTFY_TOPIC"],
        title=title,
        message=message,
        ascii_only_fn=A.ascii_only,
        log_error_fn=A.log_error,
        image_buf=image_buf,
        tags=tags,
        priority=priority,
        print_fn=print,
    )


def send_ntfy_file(filepath: str, filename: str, message: str = ""):
    import app as A

    return A.engine_send_ntfy_file(
        topic=A.CONFIG["NTFY_TOPIC"],
        filepath=filepath,
        filename=filename,
        ascii_only_fn=A.ascii_only,
        log_error_fn=A.log_error,
        message=message,
    )


def format_hourly_report_message(current_time, real_metrics: tuple, virtual_metrics: tuple, r_pnl_usd: float, v_pnl_usd: float):
    import app as A

    return A.engine_format_hourly_report_message(
        current_time=current_time,
        real_metrics=real_metrics,
        virtual_metrics=virtual_metrics,
        r_pnl_usd=r_pnl_usd,
        v_pnl_usd=v_pnl_usd,
        balance=float(A.state.balance),
        peak_balance=float(A.state.peak_balance),
        dynamic_trade_size=float(A.state.dynamic_trade_size),
        btc_atr_pct=float(A.state.btc_atr_pct),
        btc_rsi=float(A.state.btc_rsi),
        btc_adx=float(A.state.btc_adx),
        market_direction_text=str(A.state.market_direction_text),
        active_positions_count=len(A.state.active_positions),
        max_positions=int(A.CONFIG["MAX_POSITIONS"]),
        paper_positions_count=len(A.state.paper_positions),
        hourly_missed_signals=int(A.state.hourly_missed_signals),
        scan_id=int(A.state.scan_id),
    )


def create_daily_backup_zip(filepaths: list, tarih_str: str) -> str:
    import app as A

    return A.engine_create_daily_backup_zip(
        base_path=A.CONFIG["BASE_PATH"],
        filepaths=filepaths,
        tarih_str=tarih_str,
        log_error_fn=A.log_error,
    )


def upload_backup_to_github(zip_path: str, tarih_str: str) -> tuple[bool, str]:
    import app as A

    return A.engine_upload_backup_to_github(
        zip_path=zip_path,
        tarih_str=tarih_str,
        config=A.CONFIG,
        log_error_fn=A.log_error,
    )


def gunluk_dump_gonder():
    import app as A

    tarih_str = A.get_trading_day_str()
    base = A.CONFIG["BASE_PATH"]

    dosyalar = []
    for root, dirs, files in A.os.walk(base):
        files = [f for f in files if not f.endswith(".tmp") and "_old_" not in f and not f.startswith("daily_backup_")]
        for fname in sorted(files):
            dosyalar.append(A.os.path.join(root, fname))

    if not dosyalar:
        A.send_ntfy_notification(
            f"ğŸ“¦ Gunluk Dokum ({tarih_str})",
            "BASE_PATH icinde gonderilecek dosya bulunamadi.",
            tags="warning",
            priority="3",
        )
        return

    toplam_kb = round(sum(A.os.path.getsize(d) for d in dosyalar if A.os.path.exists(d)) / 1024, 1)
    zip_path = ""
    backup_note = "GitHub backup: kapali"
    try:
        zip_path = create_daily_backup_zip(dosyalar, tarih_str)
        zip_size_kb = round(A.os.path.getsize(zip_path) / 1024, 1) if A.os.path.exists(zip_path) else 0
        ok, info = upload_backup_to_github(zip_path, tarih_str)
        if ok:
            backup_note = f"GitHub backup OK -> {info} ({zip_size_kb} KB)"
        else:
            backup_note = f"GitHub backup yok ({info}) | local zip: {A.os.path.basename(zip_path)} ({zip_size_kb} KB)"
    except Exception as e:
        A.log_error("daily_zip_backup", e, tarih_str)
        backup_note = f"GitHub backup hata: {type(e).__name__} | {str(e)[:120]}"

    try:
        zip_size_kb = round(A.os.path.getsize(zip_path) / 1024, 1) if (zip_path and A.os.path.exists(zip_path)) else 0
        if zip_path and A.os.path.exists(zip_path):
            A.send_ntfy_file(
                zip_path,
                A.os.path.basename(zip_path),
                f"GUNLUK DOKUM | tarih={tarih_str} | dosya={len(dosyalar)} | toplam={toplam_kb} KB | zip={zip_size_kb} KB | {backup_note}",
            )
        else:
            A.send_ntfy_notification(
                f"GUNLUK DOKUM ({tarih_str})",
                f"ZIP olusmadi. Dosya sayisi: {len(dosyalar)}\n{backup_note}",
                tags="warning,package",
                priority="3",
            )
    except Exception as e:
        A.log_error("gunluk_dump_gonder_zip_ntfy", e, tarih_str)

