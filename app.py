# ====================== RBD-CRYPT v84.1 Quant Research Engine (Gelişmiş Durum Raporlu) ======================
import pandas as pd
import talib
import numpy as np
import requests
import os
import time
import json
import warnings
import io
import gc
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

warnings.filterwarnings('ignore')
console = Console(width=160, record=True, color_system="truecolor", force_terminal=True)

# İşletim sistemine sanal terminal tanımlaması
os.environ["TERM"] = "xterm-256color"

# ==============================================================
# 1. AYAR PANELI
# ==============================================================
CONFIG = {
    "NTFY_TOPIC": "rbd1",          
    "BASE_PATH": os.getenv("DATA_PATH", "./bot_data"), 
    "MAX_POSITIONS": 3,
    "STARTING_BALANCE": 100.0,
    "RISK_PERCENT_PER_TRADE": 25.0, 
    "MIN_TRADE_SIZE": 10.0,         
    "MAX_TRADE_SIZE": 200.0,        
    "TOP_COINS_LIMIT": 50,          
    "ST_M": 2.8,
    "RSI_PERIOD": 9,
    "RSI_LONG": 62,
    "RSI_SHORT": 38,
    "VOL_FILTER": 1.42,
    "ADX_THRESHOLD": 22,
    "MIN_ATR_PERCENT": 0.85,
    "SL_M": 1.65,
    "TP_M": 2.55,
    "COOLDOWN_MINUTES": 20,
    "MAX_HOLD_MINUTES": 30,
    "CHOP_ADX_THRESHOLD": 18,
    "BTC_VOL_THRESHOLD": 0.3,
    "TARGET_MINUTES": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],
    "MAX_LOG_SIZE_BYTES": 50_000_000 
}

FILES = {
    "LOG": os.path.join(CONFIG["BASE_PATH"], "hunter_history.csv"),
    "ACTIVE": os.path.join(CONFIG["BASE_PATH"], "active_trades.json"),
    "ALL_SIGNALS": os.path.join(CONFIG["BASE_PATH"], "all_signals.csv"),
    "STATE": os.path.join(CONFIG["BASE_PATH"], "engine_state.json")
}

if not os.path.exists(CONFIG["BASE_PATH"]): os.makedirs(CONFIG["BASE_PATH"], exist_ok=True)

# ==============================================================
# 2. SISTEM DURUMU VE DOSYA KONTROLLERİ
# ==============================================================
class HunterState:
    def __init__(self):
        self.current_coin = "Baslatiliyor..."
        self.progress_pct, self.processed_count, self.total_count = 0, 0, 0
        self.status, self.is_scanning = "BASLATILIYOR", False
        self.ranking_data = {}
        self.cooldowns = {}
        self.market_direction = "[dim]Hesaplaniyor...[/]"
        self.market_direction_text = "Hesaplanıyor..." 
        self.missed_this_scan = 0
        self.hourly_missed_signals = 0 
        self.balance = CONFIG["STARTING_BALANCE"]
        self.peak_balance = CONFIG["STARTING_BALANCE"]
        self.last_heartbeat_hour = (datetime.utcnow() + timedelta(hours=3)).hour
        self.last_dump_day = -1
        self.load_state()

    @property
    def dynamic_trade_size(self):
        size = self.balance * (CONFIG["RISK_PERCENT_PER_TRADE"] / 100.0)
        return min(max(CONFIG["MIN_TRADE_SIZE"], size), CONFIG["MAX_TRADE_SIZE"])

    def load_state(self):
        try:
            with open(FILES["ACTIVE"], 'r') as f: self.active_positions = json.load(f)
        except: self.active_positions = {}
        
        try:
            with open(FILES["STATE"], 'r') as f:
                saved_state = json.load(f)
                self.balance = saved_state.get("balance", CONFIG["STARTING_BALANCE"])
                self.peak_balance = saved_state.get("peak_balance", CONFIG["STARTING_BALANCE"])
        except: pass

    def save_state(self):
        try:
            with open(FILES["STATE"], 'w') as f:
                json.dump({"balance": self.balance, "peak_balance": self.peak_balance}, f)
            temp_file = FILES["ACTIVE"] + ".tmp"
            with open(temp_file, 'w') as f: json.dump(self.active_positions, f)
            os.replace(temp_file, FILES["ACTIVE"])
        except: pass

    def update_balance(self, pnl_percent, trade_size):
        self.balance += trade_size * (pnl_percent / 100)
        if self.balance > self.peak_balance: self.peak_balance = self.balance
        self.save_state()

state = HunterState()

def clear_terminal(): os.system('cls' if os.name == 'nt' else 'clear')
def get_tr_time(): return datetime.utcnow() + timedelta(hours=3)

def rotate_logs():
    for file_key in ["ALL_SIGNALS", "LOG"]:
        path = FILES[file_key]
        if os.path.exists(path) and os.path.getsize(path) > CONFIG["MAX_LOG_SIZE_BYTES"]:
            try: os.rename(path, path + f"_old_{int(time.time())}")
            except: pass

# --- BİLDİRİM MODÜLÜ ---
def send_ntfy_notification(title, message, image_buf=None, tags="robot", priority="3"):
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    headers = {"Title": title.encode('utf-8'), "Tags": tags, "Priority": str(priority)}
    try:
        if image_buf:
            headers["Filename"] = "chart.png"
            headers["Message"] = message.replace('\n', ' | ').encode('utf-8')
            requests.post(url, data=image_buf.getvalue(), headers=headers, timeout=10)
        else:
            requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
    except Exception as e:
        console.print(f"[bold red]❌ NTFY HATASI:[/] {e}")

def send_ntfy_file(filepath, filename, message=""):
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    headers = {"Filename": filename}
    if message:
        headers["Message"] = message.encode('utf-8')
    try:
        with open(filepath, 'rb') as f:
            requests.put(url, data=f, headers=headers, timeout=30)
    except Exception as e:
        console.print(f"[bold red]❌ NTFY DOSYA GÖNDERME HATASI:[/] {e}")

def safe_api_get(url, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200: return r.json()
            elif r.status_code == 429: time.sleep(10); continue
            else: time.sleep(2)
        except requests.exceptions.RequestException: time.sleep(3)
    return None

def get_top_futures_coins(limit=30):
    data = safe_api_get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if data:
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and '_' not in d['symbol']]
        return [p['symbol'] for p in sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]]
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def get_live_futures_data(symbol, limit=300):
    data = safe_api_get("https://fapi.binance.com/fapi/v1/klines", {"symbol": symbol, "interval": "5m", "limit": limit})
    if data:
        df = pd.DataFrame(data).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=3)
        df.set_index('timestamp', inplace=True)
        return df.apply(pd.to_numeric, errors='coerce')
    return None

def hesapla_indikatorler(df):
    c, h, l, v = df['close'].values, df['high'].values, df['low'].values, df['volume'].values
    df['RSI'] = talib.RSI(c, CONFIG["RSI_PERIOD"])
    df['ADX'] = talib.ADX(h, l, c, 14)
    df['EMA20'] = talib.EMA(c, 20)
    df['ATR_14'] = talib.ATR(h, l, c, 14)
    df['VOL_SMA_20'] = talib.SMA(v, 20)
    df['VOL_RATIO'] = np.where(df['VOL_SMA_20'] > 0, v / df['VOL_SMA_20'], 0)

    atr_st = talib.ATR(h, l, c, 10)
    hl2 = (h + l) / 2
    st_line, trend = np.zeros(len(c)), np.ones(len(c))
    for i in range(1, len(c)):
        up, dn = hl2[i] + CONFIG["ST_M"] * atr_st[i], hl2[i] - CONFIG["ST_M"] * atr_st[i]
        if c[i-1] > st_line[i-1]: st_line[i] = max(dn, st_line[i-1]); trend[i] = 1
        else: st_line[i] = min(up, st_line[i-1]); trend[i] = -1
        if trend[i] != trend[i-1]: st_line[i] = dn if trend[i] == 1 else up

    df['TREND'] = trend
    df['ST_LINE'] = st_line
    df['FLIP_LONG'] = (df['TREND'] == 1) & ((df['TREND'].shift(1) == -1) | (df['TREND'].shift(2) == -1))
    df['FLIP_SHORT'] = (df['TREND'] == -1) & ((df['TREND'].shift(1) == 1) | (df['TREND'].shift(2) == 1))
    return df

def sinyal_kontrol(row):
    is_long = row['FLIP_LONG'] and row['RSI'] > CONFIG["RSI_LONG"] and row['VOL_RATIO'] > CONFIG["VOL_FILTER"] and row['close'] > row['EMA20'] and row['ADX'] > CONFIG["ADX_THRESHOLD"]
    is_short = row['FLIP_SHORT'] and row['RSI'] < CONFIG["RSI_SHORT"] and row['VOL_RATIO'] > CONFIG["VOL_FILTER"] and row['close'] < row['EMA20'] and row['ADX'] > CONFIG["ADX_THRESHOLD"]
    if is_long: return "LONG"
    if is_short: return "SHORT"
    return None

def get_btc_trend_and_vol():
    btc_df = get_live_futures_data("BTCUSDT", 200)
    if btc_df is None or len(btc_df) < 50: return 0, 0.0
    btc_df = hesapla_indikatorler(btc_df)
    return btc_df['TREND'].iloc[-2], (btc_df['ATR_14'].iloc[-2] / btc_df['close'].iloc[-2]) * 100

def log_trade_to_csv(trade_dict):
    try:
        df = pd.read_csv(FILES["LOG"]) if os.path.exists(FILES["LOG"]) else pd.DataFrame()
        new_row = pd.DataFrame([trade_dict])
        if not new_row.empty:
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(FILES["LOG"], index=False, encoding='utf-8-sig')
    except: pass

def log_potential_signal(sym, signal_type, row, score, power_score, entered, reason=""):
    log_row = {
        'timestamp': get_tr_time().isoformat(), 'coin': sym, 'signal': signal_type,
        'score': int(score), 'power_score': round(power_score, 2), 'rsi': round(row['RSI'], 2),
        'vol_ratio': round(row['VOL_RATIO'], 2), 'adx': round(row.get('ADX', 0), 2),
        'trend': int(row['TREND']), 'atr_pct': round((row['ATR_14'] / row['close'] * 100), 2),
        'tradable': entered, 'blocked_reason': reason
    }
    pd.DataFrame([log_row]).to_csv(FILES["ALL_SIGNALS"], mode='a', header=not os.path.exists(FILES["ALL_SIGNALS"]), index=False)

def get_advanced_metrics():
    try:
        if not os.path.exists(FILES["LOG"]): return 0, 0, 0, 0.0, 0.0
        df = pd.read_csv(FILES["LOG"])
        if len(df) == 0: return 0, 0, 0, 0.0, 0.0
        wins, losses = df[df['PnL_Yuzde'] > 0], df[df['PnL_Yuzde'] <= 0]
        tot_trd, w_count = len(df), len(wins)
        gross_profit = wins['PnL_USD'].sum() if 'PnL_USD' in wins.columns else wins['PnL_Yuzde'].sum()
        gross_loss = abs(losses['PnL_USD'].sum()) if 'PnL_USD' in losses.columns else abs(losses['PnL_Yuzde'].sum())
        return tot_trd, w_count, int((w_count / tot_trd) * 100) if tot_trd > 0 else 0, round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9, ((state.peak_balance - state.balance) / state.peak_balance) * 100
    except: return 0, 0, 0, 0.0, 0.0

def create_trade_chart(df, sym, pos, is_entry=False, curr_c=None, pnl=0.0, close_reason=""):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    entry_dt = pd.to_datetime(pos.get('entry_idx_time', pos.get('full_time', get_tr_time().strftime('%Y-%m-%d %H:%M:%S'))))

    plot_df = df.tail(60).copy() if is_entry else df[df.index >= entry_dt - pd.Timedelta(minutes=150)].tail(200).copy()
    up, down = plot_df[plot_df.close >= plot_df.open], plot_df[plot_df.close < plot_df.open]
    
    ax.vlines(up.index, up.low, up.high, color='#2ecc71', linewidth=1.5, alpha=0.8)
    ax.vlines(down.index, down.low, down.high, color='#e74c3c', linewidth=1.5, alpha=0.8)
    ax.bar(up.index, up.close - up.open, 0.008, bottom=up.open, color='#2ecc71', alpha=0.9)
    ax.bar(down.index, down.open - down.close, 0.008, bottom=down.close, color='#e74c3c', alpha=0.9)

    ax.axhline(pos['entry_p'], color='#3498db', linestyle='--', alpha=0.8, label='Giris')
    ax.axhline(pos['tp'], color='#2ecc71', linestyle=':', linewidth=2, label='TP')
    ax.axhline(pos['sl'], color='#e74c3c', linestyle=':', linewidth=2, label='SL')

    if is_entry: ax.scatter(entry_dt, pos['entry_p'], color='yellow', s=150, zorder=5, edgecolors='black')
    else:
        ax.scatter(entry_dt, pos['entry_p'], color='yellow', s=120, zorder=5, edgecolors='black')
        ax.scatter(plot_df.index[-1], pos['tp'] if "KAR" in close_reason else pos['sl'], color='#2ecc71' if pnl > 0 else '#e74c3c', s=200, zorder=5, marker='X', edgecolors='white')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc='upper left', framealpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close('all')
    buf.seek(0)
    return buf

def draw_fund_dashboard():
    clear_terminal()
    strategy_text = (
        "[bold cyan]🎯 STRATEJI (v84.1 Enterprise PRO):[/] ST Flip + RSI9 + VOL>1.42x + EMA20 + ADX>22\n"
        f"[bold yellow]🏆 MAX {CONFIG['MAX_POSITIONS']} POZ | Dinamik Risk: {state.dynamic_trade_size:.1f}$ | 30dk max hold[/]\n"
        f"[bold magenta]🌍 PIYASA YONU:[/] {state.market_direction}\n"
        "[dim white]* Enterprise: Crash Guard + 429 Ban Protection + Log Rotation + Memory Cleanup + Smart Heartbeat[/]"
    )
    console.print(Panel(strategy_text, title="[bold magenta]🛠️ RBD-CRYPT v84.1 QUANT ENGINE[/]", border_style="magenta"))

    tot_trd, wins, b_wr, pf, max_dd = get_advanced_metrics()
    bal_color = "bold bright_green" if state.balance >= CONFIG["STARTING_BALANCE"] else "bold bright_red"
    
    history_text = (
        f"[dim white]Kasa Durumu:[/] [{bal_color}]${state.balance:.2f}[/] (Baslangic: ${CONFIG['STARTING_BALANCE']})\n"
        f"[dim white]Islem Basarisi:[/] [bold cyan]{wins} / {tot_trd} (%{b_wr})[/] | [dim white]Profit Factor:[/] {pf} | [dim white]Max DD:[/] %{max_dd:.2f}"
    )
    console.print(Panel(history_text, title="[bold yellow]📜 KASA VE PERFORMANS METRIKLERI[/]", border_style="yellow"))

    if len(state.active_positions) > 0:
        act_table = Table(expand=True, header_style="bold cyan", show_lines=True)
        act_table.add_column("Coin"); act_table.add_column("Yon"); act_table.add_column("Sure"); act_table.add_column("PnL (%)")

        for sym, pos in state.active_positions.items():
            curr_pnl = pos.get('curr_pnl', 0.0)
            pnl_str_c = "bright_green" if curr_pnl > 0 else "bright_red" if curr_pnl < 0 else "white"
            dur = str(get_tr_time() - datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')).split('.')[0]
            act_table.add_row(f"[bold white]{sym}[/]", "[bold bright_green]LONG[/]" if pos.get('dir') == 'LONG' else "[bold bright_red]SHORT[/]", dur, f"[bold {pnl_str_c}]{curr_pnl:.2f}%[/]")
        console.print(Panel(act_table, title=f"[bold bright_green]🟢 AKTIF ISLEMLER ({len(state.active_positions)})[/]", border_style="bright_green"))
    else:
        console.print(Panel("[dim]Su an acik pozisyon bulunmuyor...[/]", title="[bold white]⚪ AKTIF ISLEM YOK[/]", border_style="dim white"))

    if state.is_scanning: console.print(f"\n📡 [bold yellow]{state.status}[/] | Ilerleme: [bold bright_green]{state.processed_count}/{state.total_count}[/] | Coin: {state.current_coin}")
    else: console.print(f"\n📡 [bold bright_green]{state.status}[/]")

# ==============================================================
# 4. ANA KONTROL DONGUSU
# ==============================================================
def run_bot_cycle():
    state.status = "🌐 BINANCE FUTURES: Hacimli Coinler Çekiliyor..."
    draw_fund_dashboard()
    coins = get_top_futures_coins(CONFIG["TOP_COINS_LIMIT"])

    state.total_count, state.processed_count, state.is_scanning = len(coins), 0, True
    state.status = "🚀 QUANT MOTORU: VADELI PIYASA TARANIYOR (Asenkron)..."

    btc_trend_val, btc_atr_pct = get_btc_trend_and_vol()
    
    is_chop_market = False
    if btc_atr_pct < CONFIG["BTC_VOL_THRESHOLD"]:
        state.market_direction = "[bold bright_red]CHOP MARKET (BTC Dusuk Vol - SADECE VERI TOPLANIYOR)[/]"
        state.market_direction_text = "CHOP MARKET (Düşük Volatilite)"
        is_chop_market = True
    else:
        state.market_direction = "[bold bright_green]YUKSELIS (LONG)[/]" if btc_trend_val == 1 else "[bold bright_red]DUSUS (SHORT)[/]"
        state.market_direction_text = "YÜKSELİŞ (LONG)" if btc_trend_val == 1 else "DÜŞÜŞ (SHORT)"

    state.missed_this_scan = 0

    fetched_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {executor.submit(get_live_futures_data, sym, 300): sym for sym in coins}
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try: fetched_data[sym] = future.result()
            except: fetched_data[sym] = None

    for sym in coins:
        try:
            df = fetched_data.get(sym)
            state.current_coin = sym
            state.processed_count += 1
            draw_fund_dashboard()

            if df is None or len(df) < 50: continue
            df = hesapla_indikatorler(df)

            # --- A. AKTIF ISLEM KONTROLU ---
            if sym in state.active_positions:
                pos = state.active_positions[sym]
                curr_h, curr_l, curr_c = float(df['high'].iloc[-1]), float(df['low'].iloc[-1]), float(df['close'].iloc[-1])
                closed, close_reason, pnl = False, "", 0.0
                pos['curr_p'] = curr_c
                entry_p = pos.get('entry_p', curr_c)

                pos_time = datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')
                if (get_tr_time() - pos_time).total_seconds() / 60 > CONFIG["MAX_HOLD_MINUTES"]:
                    closed, close_reason = True, "TIMEOUT"
                    pnl = (curr_c - entry_p) / entry_p * 100 if pos['dir'] == 'LONG' else (entry_p - curr_c) / entry_p * 100

                if not closed:
                    if pos['dir'] == 'LONG':
                        pos['curr_pnl'] = (curr_c - entry_p) / entry_p * 100
                        if curr_h >= pos['tp']: pnl = (pos['tp'] - entry_p) / entry_p * 100; closed, close_reason = True, "KAR ALDI"
                        elif curr_l <= pos['sl']: pnl = (pos['sl'] - entry_p) / entry_p * 100; closed, close_reason = True, "STOP OLDU"
                    else:
                        pos['curr_pnl'] = (entry_p - curr_c) / entry_p * 100
                        if curr_l <= pos['tp']: pnl = (entry_p - pos['tp']) / entry_p * 100; closed, close_reason = True, "KAR ALDI"
                        elif curr_h >= pos['sl']: pnl = (entry_p - pos['sl']) / entry_p * 100; closed, close_reason = True, "STOP OLDU"

                if closed:
                    state.cooldowns[sym] = get_tr_time()
                    trade_size = pos.get('trade_size', CONFIG["MIN_TRADE_SIZE"])
                    pnl_usd = trade_size * (pnl / 100)
                    state.update_balance(pnl, trade_size) 
                    
                    trade_log = {
                        'Tarih': get_tr_time().strftime('%Y-%m-%d'), 'Giris_Saati': pos['full_time'].split(' ')[1], 'Cikis_Saati': get_tr_time().strftime('%H:%M:%S'),
                        'Coin': sym, 'Yon': pos['dir'], 'Giris_Fiyati': round(entry_p, 6), 'Cikis_Fiyati': round(curr_c, 6), 'Risk_USD': round(trade_size, 2),
                        'PnL_Yuzde': round(pnl, 2), 'PnL_USD': round(pnl_usd, 2), 'Kasa_Son_Durum': round(state.balance, 2), 'Sonuc': close_reason
                    }
                    log_trade_to_csv(trade_log)

                    try:
                        chart_buf = create_trade_chart(df, sym, pos, is_entry=False, curr_c=curr_c, pnl=pnl, close_reason=close_reason)
                        tag_emoji = "green_circle,moneybag" if pnl > 0 else "red_circle,x"
                        send_ntfy_notification(f"🔴 İŞLEM KAPANDI: {sym}", f"Sonuç: {close_reason}\nPnL: %{pnl:.2f} | Kâr/Zarar: ${pnl_usd:.2f}\nYeni Kasa: ${state.balance:.2f}", image_buf=chart_buf, tags=tag_emoji, priority="4")
                    except: pass

                    del state.active_positions[sym]
                    state.save_state()

            # --- B. YENI SINYAL ---
            row_closed = df.iloc[-2]
            signal = sinyal_kontrol(row_closed)
            entered, reason = False, ""

            if signal:
                if is_chop_market: reason = "CHOP_MARKET"
                elif signal == "LONG" and btc_trend_val != 1: reason = "BTC_TREND_KOTU"
                elif signal == "SHORT" and btc_trend_val != -1: reason = "BTC_TREND_KOTU"
                elif sym in state.active_positions: reason = "ALREADY_IN"
                elif sym in state.cooldowns and (get_tr_time() - state.cooldowns[sym]).total_seconds() / 60 < CONFIG["COOLDOWN_MINUTES"]: reason = "COOLDOWN"
                elif len(state.active_positions) >= CONFIG["MAX_POSITIONS"]: reason = "MAX_POS"
                elif row_closed['ADX'] < CONFIG["CHOP_ADX_THRESHOLD"]: reason = "LOW_ADX"
                else:
                    entry_p = float(row_closed['close'])
                    atr_val = float(row_closed['ATR_14'])
                    sl_p = entry_p - (CONFIG["SL_M"] * atr_val) if signal == "LONG" else entry_p + (CONFIG["SL_M"] * atr_val)
                    tp_p = entry_p + (CONFIG["TP_M"] * atr_val) if signal == "LONG" else entry_p - (CONFIG["TP_M"] * atr_val)
                    t_size = state.dynamic_trade_size

                    state.active_positions[sym] = {
                        'dir': signal, 'entry_p': entry_p, 'sl': sl_p, 'tp': tp_p,
                        'full_time': get_tr_time().strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_idx_time': str(row_closed.name), 'curr_pnl': 0.0, 'curr_p': float(df['close'].iloc[-1]), 'trade_size': t_size 
                    }
                    state.save_state()
                    entered = True
                    
                    try: 
                        chart_buf = create_trade_chart(df, sym, state.active_positions[sym], is_entry=True)
                        send_ntfy_notification(f"🟢 YENİ İŞLEM: {sym}", f"Yön: {signal}\nFiyat: {entry_p:.5f}\nSL: {sl_p:.5f} | TP: {tp_p:.5f}\nRisk: ${t_size:.2f}", image_buf=chart_buf, tags="chart_with_upwards_trend", priority="4")
                    except: pass
                
                if not entered: 
                    state.missed_this_scan += 1
                    state.hourly_missed_signals += 1 
                
                log_potential_signal(sym, signal, row_closed, 0, 0, entered, reason)

        except Exception as e: pass

    del fetched_data
    gc.collect()

    state.is_scanning = False
    rotate_logs()
    
    current_time = get_tr_time()
    current_hour = current_time.hour
    
    # ⏱️ SAATLİK RAPOR
    if current_hour != state.last_heartbeat_hour:
        state.last_heartbeat_hour = current_hour
        hb_msg = (
            f"💵 Kasa: ${state.balance:.2f} (Tepe: ${state.peak_balance:.2f})\n"
            f"🌍 Piyasa Yönü: {state.market_direction_text}\n"
            f"⛔ 1 Saatte Reddedilen Sinyal: {state.hourly_missed_signals} adet\n"
            f"📈 Açık İşlem: {len(state.active_positions)}/{CONFIG['MAX_POSITIONS']}\n"
            f"Sistem stabil, disiplin bozulmuyor patron."
        )
        send_ntfy_notification(f"⏱️ Saatlik Rapor ({current_time.strftime('%H:00')})", hb_msg, tags="clipboard,bar_chart", priority="3")
        state.hourly_missed_signals = 0 

    # 💾 GÜNLÜK TÜM DOSYALARI DÖKÜMLEME (23:56 - 23:59 arası günde 1 kere)
    current_day = current_time.day
    if current_hour == 23 and current_time.minute >= 56 and state.last_dump_day != current_day:
        state.last_dump_day = current_day
        send_ntfy_notification("📦 GÜNLÜK DOSYA DÖKÜMÜ", "Sistemdeki tüm log ve kayıt dosyaları gönderiliyor...", tags="package", priority="4")
        for key, path in FILES.items():
            if os.path.exists(path):
                send_ntfy_file(path, os.path.basename(path), f"Günün Dosyası: {key}")

    now = get_tr_time()
    target = now.replace(second=0, microsecond=0)
    next_min = next((m for m in CONFIG["TARGET_MINUTES"] if m > now.minute), CONFIG["TARGET_MINUTES"][0])
    if next_min == CONFIG["TARGET_MINUTES"][0]: target += timedelta(hours=1); target = target.replace(minute=next_min)
    else: target = target.replace(minute=next_min)

    state.status = f"💤 SENKRON BEKLEME (Sonraki Tarama: [bold bright_green]{target.strftime('%H:%M:%S')}[/])"
    draw_fund_dashboard()

    sleep_sec = (target - now).total_seconds()
    if sleep_sec > 0: time.sleep(sleep_sec)


# ==============================================================
# 5. GLOBAL CRASH GUARD
# ==============================================================
if __name__ == "__main__":
    start_msg = f"💵 Güncel Kasa: ${state.balance:.2f}\n🛡️ Global Crash Guard Aktif\n💾 Log Rotation & RAM Koruması Devrede\nBot production modunda taranıyor!"
    send_ntfy_notification("🚀 v84.1 BAŞLATILDI", start_msg, tags="rocket,shield", priority="4")
    
    while True:
        try:
            run_bot_cycle()
        except Exception as e:
            error_msg = f"Sistem Hata Aldı ve Çöktü!\nHata: {str(e)[:150]}\n\n30 Saniye içinde kendini onarıp tekrar başlayacak."
            console.print(f"\n[bold red on white] 🚨 CRITICAL ERROR: {e} [/]")
            send_ntfy_notification("🚨 SİSTEM ÇÖKTÜ (RESTART ATILIYOR)", error_msg, tags="rotating_light,warning", priority="5")
            time.sleep(30)