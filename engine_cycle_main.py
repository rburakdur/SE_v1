import concurrent.futures

from engine_cycle_active import process_active_real_position
from engine_cycle_post import finalize_cycle_and_sleep
from engine_cycle_signal import process_signal_for_coin


def run_bot_cycle():
    import app as A

    A.state.scan_id += 1
    A.state.save_state()

    A.state.status = "ğŸŒ BINANCE FUTURES: Hacimli Coinler cekiliyor..."
    A.draw_fund_dashboard()
    scan_limit = int(max(A.CONFIG["TOP_COINS_LIMIT"], A.CONFIG.get("CANDIDATE_TOP_COINS_LIMIT", A.CONFIG["TOP_COINS_LIMIT"])))
    coins = A.get_top_futures_coins(scan_limit)

    btc_ctx = A.get_btc_context()
    A.state.btc_atr_pct = btc_ctx["atr_pct"]
    A.state.btc_rsi = btc_ctx["rsi"]
    A.state.btc_adx = btc_ctx["adx"]
    A.state.btc_vol_ratio = btc_ctx["vol_ratio"]

    A.state.is_chop_market = btc_ctx["atr_pct"] < A.CONFIG["BTC_VOL_THRESHOLD"]
    if A.state.is_chop_market:
        A.state.market_direction_text = "CHOP MARKET (Dusuk Volatilite)"
    else:
        A.state.market_direction_text = "YUKSELIS (LONG)" if btc_ctx["trend"] == 1 else "DUSUS (SHORT)"

    A.log_market_context(btc_ctx, len(coins), len(A.state.active_positions))

    A.state.total_count = len(coins)
    A.state.processed_count = 0
    A.state.is_scanning = True
    A.state.missed_this_scan = 0
    A.state.status = "ğŸš€ QUANT MOTORU: VADELI PIYASA TARANIYOR (Asenkron)..."

    fetched_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {executor.submit(A.get_live_futures_data, sym, 300): sym for sym in coins}
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                fetched_data[sym] = future.result()
            except Exception as e:
                A.log_error("fetch_worker", e, sym)
                fetched_data[sym] = None

    for sym in coins:
        try:
            df = fetched_data.get(sym)
            A.state.current_coin = sym
            A.state.processed_count += 1
            if A.state.processed_count % 10 == 1 or A.state.processed_count == A.state.total_count:
                A.draw_fund_dashboard()

            if df is None or len(df) < 50:
                continue

            try:
                df = A.hesapla_indikatorler(df, sym)
            except Exception as e:
                try:
                    dtypes_str = ",".join(f"{c}:{df[c].dtype}" for c in ["open", "high", "low", "close", "volume"] if c in df.columns)
                    extra = f"{sym} | {dtypes_str}"
                except Exception:
                    extra = sym
                A.log_error("hesapla_indikatorler", e, extra)
                continue

            if process_active_real_position(sym, df, btc_ctx):
                continue

            if sym in A.state.paper_positions:
                A.update_paper_position_for_symbol(sym, df, btc_ctx)

            process_signal_for_coin(sym, df, btc_ctx)
        except Exception as e:
            A.log_error("coin_loop", e, sym)
            continue

    finalize_cycle_and_sleep(fetched_data)

