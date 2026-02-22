# ====================== RBD-CRYPT v83.0 Quant Research Engine (Tam Otomasyon & Raporlama) ======================
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
import os
# İşletim sistemine sahte bir terminal veriyoruz
os.environ["TERM"] = "xterm-256color"

warnings.filterwarnings('ignore')
console = Console(width=160, record=True, color_system="truecolor", force_terminal=True)

# ==============================================================
# 1. AYAR PANELI (Lokal VSCode Uyumlu)
# ==============================================================
CONFIG = {
    "NTFY_TOPIC": "rbd1",          # Bildirim Kanalı (Ntfy.sh)
    "BASE_PATH": './bot_data',     # VSCode için lokal klasör
    "MAX_POSITIONS": 3,
    "STARTING_BALANCE": 100.0,
    
    # --- DİNAMİK RİSK YÖNETİMİ ---
    "RISK_PERCENT_PER_TRADE": 25.0, 
    "MIN_TRADE_SIZE": 10.0,         
    "MAX_TRADE_SIZE": 200.0,        
    
    "TOP_COINS_LIMIT": 30,          
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
    "TARGET_MINUTES": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56]
}

FILES = {
    "LOG": os.path.join(CONFIG["BASE_PATH"], "hunter_history.csv"),
    "ACTIVE": os.path.join(CONFIG["BASE_PATH"], "active_trades.json"),
    "ALL_SIGNALS": os.path.join(CONFIG["BASE_PATH"], "all_signals.csv"),
    "STATE": os.path.join(CONFIG["BASE_PATH"], "engine_state.json")
}

if not os.path.exists(CONFIG["BASE_PATH"]): os.makedirs(CONFIG["BASE_PATH"], exist_ok=True)

# ==============================================================
# 2. SISTEM DURUMU VE VERI MODULLERI
# ==============================================================
class HunterState:
    def __init__(self):
        self.current_coin = "Baslatiliyor..."
        self.progress_pct, self.processed_count, self.total_count = 0, 0, 0
        self.status, self.is_scanning = "BASLATILIYOR", False
        self.ranking_data = {}
        self.cooldowns = {}
        self.market_direction = "[dim]Hesaplaniyor...[/]"
        self.missed_this_scan = 0
        self.balance = CONFIG["STARTING_BALANCE"]
        self.peak_balance = CONFIG["STARTING_BALANCE"]
        self.last_heartbeat_hour = (datetime.utcnow() + timedelta(hours=3)).hour
        
        self.load_state()

    @property
    def dynamic_trade_size(self):
        size = self.balance * (CONFIG["RISK_PERCENT_PER_TRADE"] / 100.0)
        size = max(CONFIG["MIN_TRADE_SIZE"], size)
        size = min(CONFIG["MAX_TRADE_SIZE"], size)
        return size

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
        pnl_usd = trade_size * (pnl_percent / 100)
        self.balance += pnl_usd
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.save_state()

state = HunterState()

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_tr_time():
    return datetime.utcnow() + timedelta(hours=3)

# --- BİLDİRİM MODÜLÜ (GELİŞMİŞ VE HATASIZ) ---
def send_ntfy_notification(title, message, image_buf=None, tags="robot", priority="3"):
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    
    # Python'ın requests kütüphanesi HTTP Header'larda Emoji ve Türkçe karakter sevmez.
    # O yüzden title'ı doğrudan utf-8 byte'a çeviriyoruz (.encode('utf-8')).
    headers = {
        "Title": title.encode('utf-8'),
        "Tags": tags,
        "Priority": str(priority)
    }
    
    try:
        if image_buf:
            headers["Filename"] = "chart.png"
            # Resim atarken mesaj metni de Header'a girdiği için, yeni satırları (\n) ayırıcıya (|) çevirip encode ediyoruz.
            clean_msg = message.replace('\n', ' | ')
            headers["Message"] = clean_msg.encode('utf-8')
            
            requests.post(url, data=image_buf.getvalue(), headers=headers, timeout=10)
        else:
            requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
            
    except Exception as e:
        # Eğer internet koparsa veya bir sorun çıkarsa eskisi gibi sessiz kalmayacak, ekrana basacak!
        console.print(f"[bold red]❌ NTFY BİLDİRİM GÖNDERİLEMEDİ:[/] {e}")

# --- API VE VERİ MODULLERI ---
def safe_api_get(url, params=None, retries=3):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200: return r.json()
        except:
            time.sleep(1)
    return None

def get_top_futures_coins(limit=30):
    data = safe_api_get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if data:
        usdt_pairs = [d for d in data if d['symbol'].endswith('USDT') and '_' not in d['symbol']]
        sorted_pairs = sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)
        return [p['symbol'] for p in sorted_pairs[:limit]]
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
    trend = btc_df['TREND'].iloc[-2]
    atr_pct = (btc_df['ATR_14'].iloc[-2] / btc_df['close'].iloc[-2]) * 100
    return trend, atr_pct

# --- LOGLAMA VE METRIKLER ---
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
        'timestamp': get_tr_time().isoformat(),
        'coin': sym,
        'signal': signal_type,
        'score': int(score),
        'power_score': round(power_score, 2),
        'rsi': round(row['RSI'], 2),
        'vol_ratio': round(row['VOL_RATIO'], 2),
        'adx': round(row.get('ADX', 0), 2),
        'trend': int(row['TREND']),
        'atr_pct': round((row['ATR_14'] / row['close'] * 100), 2),
        'tradable': entered,
        'blocked_reason': reason
    }
    pd.DataFrame([log_row]).to_csv(FILES["ALL_SIGNALS"], mode='a', header=not os.path.exists(FILES["ALL_SIGNALS"]), index=False)

def get_advanced_metrics():
    try:
        if not os.path.exists(FILES["LOG"]): return 0, 0, 0, 0.0, 0.0
        df = pd.read_csv(FILES["LOG"])
        if len(df) == 0: return 0, 0, 0, 0.0, 0.0
        
        wins = df[df['PnL_Yuzde'] > 0]
        losses = df[df['PnL_Yuzde'] <= 0]
        
        tot_trd = len(df)
        w_count = len(wins)
        win_rate = int((w_count / tot_trd) * 100) if tot_trd > 0 else 0
        
        gross_profit = wins['PnL_USD'].sum() if 'PnL_USD' in wins.columns else wins['PnL_Yuzde'].sum()
        gross_loss = abs(losses['PnL_USD'].sum()) if 'PnL_USD' in losses.columns else abs(losses['PnL_Yuzde'].sum())
        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9
        
        current_dd = ((state.peak_balance - state.balance) / state.peak_balance) * 100
        
        return tot_trd, w_count, win_rate, profit_factor, current_dd
    except: return 0, 0, 0, 0.0, 0.0

# ==============================================================
# 3. GORSEL MOTOR (Dashboard & Grafik)
# ==============================================================
def create_trade_chart(df, sym, pos, is_entry=False, curr_c=None, pnl=0.0, close_reason=""):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    entry_dt = pd.to_datetime(pos.get('entry_idx_time', pos.get('full_time', get_tr_time().strftime('%Y-%m-%d %H:%M:%S'))))

    if is_entry: plot_df = df.tail(60).copy()
    else:
        start_dt = entry_dt - pd.Timedelta(minutes=150)
        plot_df = df[df.index >= start_dt].copy()
        if len(plot_df) > 200: plot_df = plot_df.tail(200)

    up, down = plot_df[plot_df.close >= plot_df.open], plot_df[plot_df.close < plot_df.open]
    ax.vlines(up.index, up.low, up.high, color='#2ecc71', linewidth=1.5, alpha=0.8)
    ax.vlines(down.index, down.low, down.high, color='#e74c3c', linewidth=1.5, alpha=0.8)
    width = 0.008
    ax.bar(up.index, up.close - up.open, width, bottom=up.open, color='#2ecc71', alpha=0.9)
    ax.bar(down.index, down.open - down.close, width, bottom=down.close, color='#e74c3c', alpha=0.9)

    ax.axhline(pos['entry_p'], color='#3498db', linestyle='--', alpha=0.8, label='Giris')
    ax.axhline(pos['tp'], color='#2ecc71', linestyle=':', linewidth=2, label='TP')
    ax.axhline(pos['sl'], color='#e74c3c', linestyle=':', linewidth=2, label='SL')

    if is_entry: ax.scatter(entry_dt, pos['entry_p'], color='yellow', s=150, zorder=5, edgecolors='black')
    else:
        ax.scatter(entry_dt, pos['entry_p'], color='yellow', s=120, zorder=5, edgecolors='black')
        exit_price = pos['tp'] if "KAR" in close_reason else pos['sl']
        ax.scatter(plot_df.index[-1], exit_price, color='#2ecc71' if pnl > 0 else '#e74c3c', s=200, zorder=5, marker='X', edgecolors='white')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc='upper left', framealpha=0.3)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    plt.close(fig); gc.collect()
    buf.seek(0)
    return buf

def draw_fund_dashboard():
    clear_terminal()
    
    strategy_text = (
        "[bold cyan]🎯 STRATEJI (v83.0 Quant Engine PRO):[/] ST Flip + RSI9 + VOL>1.42x + EMA20 + ADX>22\n"
        f"[bold yellow]🏆 MAX {CONFIG['MAX_POSITIONS']} POZ | Dinamik Risk: {state.dynamic_trade_size:.1f}$ (Kasa %{CONFIG['RISK_PERCENT_PER_TRADE']}) | 30dk max hold[/]\n"
        f"[bold magenta]🌍 PIYASA YONU:[/] {state.market_direction}\n"
        "[dim white]* Kurumsal: Otonom Hacim Taramasi + Tam Sinyal Loglama + Gerçek Kasa Takibi + Multi-Threading[/]"
    )
    console.print(Panel(strategy_text, title="[bold magenta]🛠️ RBD-CRYPT v83.0 QUANT RESEARCH ENGINE[/]", border_style="magenta"))

    tot_trd, wins, b_wr, pf, max_dd = get_advanced_metrics()
    
    bal_color = "bold bright_green" if state.balance >= CONFIG["STARTING_BALANCE"] else "bold bright_red"
    pf_color = "bold bright_green" if pf >= 1.5 else ("bold yellow" if pf >= 1.0 else "bold bright_red")
    
    history_text = (
        f"[dim white]Kasa Durumu:[/] [{bal_color}]${state.balance:.2f}[/] (Baslangic: ${CONFIG['STARTING_BALANCE']})\n"
        f"[dim white]Islem Basarisi:[/] [bold cyan]{wins} / {tot_trd} (%{b_wr})[/] | [dim white]Profit Factor:[/] [{pf_color}]{pf}[/] | [dim white]Max Drawdown:[/] [bold red]%{max_dd:.2f}[/]"
    )
    console.print(Panel(history_text, title="[bold yellow]📜 KASA VE PERFORMANS METRIKLERI[/]", border_style="yellow"))

    if len(state.active_positions) > 0:
        act_table = Table(expand=True, header_style="bold cyan", show_lines=True)
        act_table.add_column("Coin", justify="center")
        act_table.add_column("Yon", justify="center")
        act_table.add_column("Sure", justify="center")
        act_table.add_column("Giris Fiyati", justify="center")
        act_table.add_column("Anlik Fiyat", justify="center")
        act_table.add_column("PnL (%)", justify="center")
        act_table.add_column("PnL ($)", justify="center")

        for sym, pos in state.active_positions.items():
            dir_c = "[bold bright_green]LONG[/]" if pos.get('dir') == 'LONG' else "[bold bright_red]SHORT[/]"
            curr_pnl = pos.get('curr_pnl', 0.0)
            
            t_size = pos.get('trade_size', CONFIG["MIN_TRADE_SIZE"])
            curr_usd = t_size * (curr_pnl / 100)
            
            pnl_str_c = "bright_green" if curr_pnl > 0 else ("bright_red" if curr_pnl < 0 else "white")
            
            dur = "--"
            try: dur = str(get_tr_time() - datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')).split('.')[0]
            except: pass
            
            act_table.add_row(
                f"[bold white]{sym}[/]", dir_c, dur,
                f"{pos['entry_p']:.6f}", f"{pos.get('curr_p', 0):.6f}",
                f"[bold {pnl_str_c}]{curr_pnl:.2f}%[/]", f"[bold {pnl_str_c}]${curr_usd:.2f}[/]"
            )
        console.print(Panel(act_table, title=f"[bold bright_green]🟢 AKTIF ISLEMLER ({len(state.active_positions)} Adet)[/]", border_style="bright_green"))
    else:
        console.print(Panel("[dim]Su an acik pozisyon bulunmuyor...[/]", title="[bold white]⚪ AKTIF ISLEM YOK[/]", border_style="dim white"))

    if state.is_scanning:
        bar = "█" * int(state.progress_pct/5) + "░" * (20 - int(state.progress_pct/5))
        console.print(f"\n📡 [bold yellow]{state.status}[/] | Ilerleme: [bold bright_green]{state.processed_count}/{state.total_count} (%{state.progress_pct})[/] [{bar}]")
        console.print(f"[bold cyan]🔍 Analiz Ediliyor:[/] [bold white]{state.current_coin}[/]")
    else:
        console.print(f"\n📡 [bold bright_green]{state.status}[/]")
        console.print(f"\n📉 Bu taramada [bold red]{state.missed_this_scan}[/] potansiyel sinyal MISSED (Tümü CSV'ye Loglandı)")

# ==============================================================
# 4. ANA KONTROL DONGUSU
# ==============================================================
if __name__ == "__main__":
    
    # 🚀 Bot Açılış Bildirimi (Gözün arkada kalmasın)
    start_msg = f"💵 Güncel Kasa: ${state.balance:.2f}\n⚖️ Hedef Risk: %{CONFIG['RISK_PERCENT_PER_TRADE']} (Dinamik)\n📉 Bütün hareketler ve missed sinyaller loglanıyor.\nBot göreve hazır, sen işine bak patron!"
    send_ntfy_notification("🚀 RBD-CRYPT v83.0 Başlatıldı", start_msg, tags="rocket,white_check_mark", priority="4")
    
    while True:
        loop_start_time = get_tr_time()

        state.status = "🌐 BINANCE FUTURES: Hacimli Coinler Çekiliyor..."
        draw_fund_dashboard()
        coins = get_top_futures_coins(CONFIG["TOP_COINS_LIMIT"])

        state.total_count, state.processed_count, state.is_scanning = len(coins), 0, True
        state.status = "🚀 QUANT MOTORU: VADELI PIYASA TARANIYOR (Asenkron)..."

        btc_trend_val, btc_atr_pct = get_btc_trend_and_vol()
        
        is_chop_market = False
        if btc_atr_pct < CONFIG["BTC_VOL_THRESHOLD"]:
            state.market_direction = "[bold bright_red]CHOP MARKET (BTC Dusuk Vol - SADECE VERI TOPLANIYOR)[/]"
            is_chop_market = True
        else:
            state.market_direction = "[bold bright_green]YUKSELIS (LONG)[/]" if btc_trend_val == 1 else "[bold bright_red]DUSUS (SHORT)[/]"

        state.missed_this_scan = 0

        # --- MULTI-THREADING İLE HIZLI VERİ ÇEKİMİ ---
        fetched_data = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_sym = {executor.submit(get_live_futures_data, sym, 300): sym for sym in coins}
            for future in concurrent.futures.as_completed(future_to_sym):
                sym = future_to_sym[future]
                try: fetched_data[sym] = future.result()
                except: fetched_data[sym] = None

        # Çekilen veriler üzerinden analizi yap
        for sym in coins:
            try:
                df = fetched_data.get(sym)
                state.current_coin = sym
                state.processed_count += 1
                state.progress_pct = int((state.processed_count / state.total_count) * 100)
                draw_fund_dashboard()

                if df is None or len(df) < 50: continue

                df = hesapla_indikatorler(df)

                # --- A. AKTIF ISLEM KONTROLU + TIMEOUT ---
                if sym in state.active_positions:
                    pos = state.active_positions[sym]
                    curr_h, curr_l, curr_c = float(df['high'].iloc[-1]), float(df['low'].iloc[-1]), float(df['close'].iloc[-1])
                    closed, close_reason, pnl = False, "", 0.0
                    pos['curr_p'] = curr_c
                    entry_p = pos.get('entry_p', curr_c)

                    pos_time = datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')
                    hold_min = (get_tr_time() - pos_time).total_seconds() / 60
                    if hold_min > CONFIG["MAX_HOLD_MINUTES"]:
                        closed = True
                        close_reason = "TIMEOUT"
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
                        
                        # KASAYI GÜNCELLE VE YENİ BAKIYEYI LOGLA
                        state.update_balance(pnl, trade_size) 
                        
                        # 📝 DETAYLI TRADE LOGLAMA (Artık yatırılan Risk_USD ve Kasa_Son_Durum da var)
                        trade_log = {
                            'Tarih': get_tr_time().strftime('%Y-%m-%d'),
                            'Giris_Saati': pos['full_time'].split(' ')[1],
                            'Cikis_Saati': get_tr_time().strftime('%H:%M:%S'),
                            'Coin': sym, 'Yon': pos['dir'],
                            'Giris_Fiyati': round(entry_p, 6), 'Cikis_Fiyati': round(curr_c, 6),
                            'Risk_USD': round(trade_size, 2),
                            'PnL_Yuzde': round(pnl, 2), 'PnL_USD': round(pnl_usd, 2), 
                            'Kasa_Son_Durum': round(state.balance, 2),
                            'Sonuc': close_reason
                        }
                        log_trade_to_csv(trade_log)

                        # 📱 NTFY ÇIKIŞ BİLDİRİMİ
                        try:
                            chart_buf = create_trade_chart(df, sym, pos, is_entry=False, curr_c=curr_c, pnl=pnl, close_reason=close_reason)
                            tag_emoji = "green_circle,moneybag" if pnl > 0 else "red_circle,x"
                            msg = f"Sonuç: {close_reason}\nPnL: %{pnl:.2f} | Kâr/Zarar: ${pnl_usd:.2f}\nYeni Kasa: ${state.balance:.2f}"
                            send_ntfy_notification(f"🔴 İŞLEM KAPANDI: {sym}", msg, image_buf=chart_buf, tags=tag_emoji, priority="4")
                        except: pass

                        del state.active_positions[sym]
                        state.save_state()

                # --- B. YENI SINYAL + AGRESİF LOGLAMA ---
                closed_idx = -2
                row_closed = df.iloc[closed_idx]
                curr_trend, v_ratio, curr_rsi, curr_adx, curr_atr_pct, curr_c_closed, curr_ema = row_closed['TREND'], row_closed['VOL_RATIO'], row_closed['RSI'], row_closed['ADX'], (row_closed['ATR_14'] / row_closed['close']) * 100, row_closed['close'], row_closed['EMA20']

                v_ok = v_ratio > CONFIG["VOL_FILTER"]
                adx_ok = curr_adx > CONFIG["ADX_THRESHOLD"]
                atr_ok = curr_atr_pct > CONFIG["MIN_ATR_PERCENT"]
                
                if curr_trend == 1:
                    f_ok = row_closed['FLIP_LONG']
                    r_ok = curr_rsi > CONFIG["RSI_LONG"]
                    e_ok = curr_c_closed > curr_ema
                else:
                    f_ok = row_closed['FLIP_SHORT']
                    r_ok = curr_rsi < CONFIG["RSI_SHORT"]
                    e_ok = curr_c_closed < curr_ema

                score = sum([f_ok, r_ok, v_ok, e_ok, adx_ok, atr_ok])
                if curr_adx < CONFIG["CHOP_ADX_THRESHOLD"]: score = 0
                power_score = (score * 10000) + (v_ratio * 100)

                signal = sinyal_kontrol(row_closed)

                entered = False
                reason = ""

                if signal:
                    if is_chop_market: reason = "CHOP_MARKET"
                    elif signal == "LONG" and btc_trend_val != 1: reason = "BTC_TREND_KOTU"
                    elif signal == "SHORT" and btc_trend_val != -1: reason = "BTC_TREND_KOTU"
                    elif sym in state.active_positions: reason = "ALREADY_IN"
                    elif sym in state.cooldowns and (get_tr_time() - state.cooldowns[sym]).total_seconds() / 60 < CONFIG["COOLDOWN_MINUTES"]: reason = "COOLDOWN"
                    elif len(state.active_positions) >= CONFIG["MAX_POSITIONS"]: reason = "MAX_POS"
                    elif score < 5: reason = "LOW_SCORE"
                    else:
                        entry_p = float(row_closed['close'])
                        atr_val = float(row_closed['ATR_14'])
                        sl_p = entry_p - (CONFIG["SL_M"] * atr_val) if signal == "LONG" else entry_p + (CONFIG["SL_M"] * atr_val)
                        tp_p = entry_p + (CONFIG["TP_M"] * atr_val) if signal == "LONG" else entry_p - (CONFIG["TP_M"] * atr_val)
                        
                        t_size = state.dynamic_trade_size

                        state.active_positions[sym] = {
                            'dir': signal, 'entry_p': entry_p, 'sl': sl_p, 'tp': tp_p,
                            'full_time': get_tr_time().strftime('%Y-%m-%d %H:%M:%S'),
                            'entry_idx_time': str(row_closed.name),
                            'curr_pnl': 0.0, 'curr_p': float(df['close'].iloc[-1]),
                            'trade_size': t_size 
                        }
                        state.save_state()
                        entered = True
                        
                        # 📱 NTFY GİRİŞ BİLDİRİMİ
                        try: 
                            chart_buf = create_trade_chart(df, sym, state.active_positions[sym], is_entry=True)
                            msg = f"Yön: {signal}\nFiyat: {entry_p:.5f}\nSL: {sl_p:.5f} | TP: {tp_p:.5f}\nRisk Edilen: ${t_size:.2f}\nRSI: {curr_rsi:.1f} | Hacim Oranı: {v_ratio:.2f}x"
                            send_ntfy_notification(f"🟢 YENİ İŞLEM: {sym}", msg, image_buf=chart_buf, tags="chart_with_upwards_trend", priority="4")
                        except: pass
                    
                    if not entered: 
                        state.missed_this_scan += 1
                    
                    # HER HALTI KAYDEDEN FONKSIYON: Sinyal oluşmuşsa (girmese bile) yazar.
                    log_potential_signal(sym, signal, row_closed, score, power_score, entered, reason)

            except Exception as e: pass
            time.sleep(0.02) 

        # Temizlik
        gc.collect()
        state.is_scanning = False
        
        # ⏱️ SAATLİK HEARTBEAT (KALP ATIŞI) KONTROLÜ
        current_hour = get_tr_time().hour
        if current_hour != state.last_heartbeat_hour:
            state.last_heartbeat_hour = current_hour
            hb_msg = f"💵 Güncel Kasa: ${state.balance:.2f}\n📈 Açık İşlem Sayısı: {len(state.active_positions)}\nSistem tıkır tıkır işliyor, sorun yok."
            send_ntfy_notification(f"⏱️ Saatlik Özet Raporu ({get_tr_time().strftime('%H:00')})", hb_msg, tags="hourglass,clipboard", priority="3")

        # Senkron Bekleme Ayarı
        now = get_tr_time()
        target = now.replace(second=0, microsecond=0)
        next_min = next((m for m in CONFIG["TARGET_MINUTES"] if m > now.minute), CONFIG["TARGET_MINUTES"][0])
        if next_min == CONFIG["TARGET_MINUTES"][0]:
            target += timedelta(hours=1)
            target = target.replace(minute=next_min)
        else:
            target = target.replace(minute=next_min)

        state.status = f"💤 SENKRON BEKLEME (Sonraki Tarama: [bold bright_green]{target.strftime('%H:%M:%S')}[/])"
        draw_fund_dashboard()

        sleep_sec = (target - now).total_seconds()
        if sleep_sec > 0: time.sleep(sleep_sec)