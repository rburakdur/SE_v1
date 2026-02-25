def format_hourly_report_message(
    current_time,
    real_metrics: tuple,
    virtual_metrics: tuple,
    r_pnl_usd: float,
    v_pnl_usd: float,
    *,
    balance: float,
    peak_balance: float,
    dynamic_trade_size: float,
    btc_atr_pct: float,
    btc_rsi: float,
    btc_adx: float,
    market_direction_text: str,
    active_positions_count: int,
    max_positions: int,
    paper_positions_count: int,
    hourly_missed_signals: int,
    scan_id: int,
    title_prefix: str = "RBD-CRYPT v87",
) -> tuple[str, str]:
    tot_trd, wins, b_wr, pf, max_dd = real_metrics
    v_tot, v_wins, v_wr, v_pf, _ = virtual_metrics
    title = f"{title_prefix} | Saatlik {current_time.strftime('%H:00')}"
    body = (
        f"Kasa ${balance:.2f} | Tepe ${peak_balance:.2f} | Poz ${dynamic_trade_size:.1f}\n"
        f"Gercek {wins}/{tot_trd} | %{b_wr} | PnL ${r_pnl_usd:.2f} | PF {pf} | DD %{max_dd:.2f}\n"
        f"Sanal {v_wins}/{v_tot} | %{v_wr} | PnL ${v_pnl_usd:.2f} | PF {v_pf}\n"
        f"BTC ATR {btc_atr_pct:.3f} | RSI {btc_rsi:.1f} | ADX {btc_adx:.1f}\n"
        f"Piyasa: {market_direction_text}\n"
        f"Acik: Gercek {active_positions_count}/{max_positions} | Sanal {paper_positions_count}\n"
        f"Red(1s): {hourly_missed_signals} | Scan: {scan_id}"
    )
    return title, body


def build_status_message(
    *,
    balance: float,
    peak_balance: float,
    max_positions: int,
    active_positions: dict,
    real_stats: dict,
    virtual_stats: dict,
    market_direction_text: str,
    scan_id: int,
) -> str:
    open_count = len(active_positions)
    position_lines = []
    for sym, pos in active_positions.items():
        pnl = float(pos.get("curr_pnl", 0.0) or 0.0)
        position_lines.append(f"  - {sym} {pos.get('dir', '?')} | PnL: %{pnl:.2f}")

    positions_text = ("\n".join(position_lines) + "\n") if position_lines else ""
    return (
        f"Kasa: ${balance:.2f} (Tepe: ${peak_balance:.2f})\n"
        f"Acik Islem: {open_count}/{max_positions}\n"
        f"{positions_text}"
        f"Basari: {real_stats.get('wins', 0)}/{real_stats.get('total_trades', 0)} "
        f"(%{real_stats.get('win_rate', 0)}) | PF: {real_stats.get('pf', 0.0)}\n"
        f"Toplam PnL: Gercek ${float(real_stats.get('net_pnl_usd', 0.0)):.2f} | "
        f"Sanal ${float(virtual_stats.get('net_pnl_usd', 0.0)):.2f}\n"
        f"Piyasa: {market_direction_text}\n"
        f"Scan ID: {scan_id}"
    )

