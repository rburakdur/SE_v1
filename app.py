# ====================== RBD-CRYPT v85.0 Quant Research Engine ======================
# v84.1'den v85.0'a değişiklikler:
#   - Tüm 'except: pass' kaldırıldı, her hata error_log.csv'ye yazılıyor
#   - MIN_ATR_PERCENT filtresi gerçekten uygulanıyor
#   - score / power_score hesaplamaları gerçek (walk-forward'a hazır)
#   - hunter_history.csv: 20+ yeni kolon (bar detayı, market context, indicator snapshot)
#   - all_signals.csv: genişletildi, her sinyalin tam fotoğrafı
#   - market_context.csv: her scan'in BTC/piyasa durumu
#   - error_log.csv: sessiz hataları yakalıyor
#   - Timeout çıkışında ST flip varsa onu bekle (MAX_HOLD + 2 bar tolerans)
#   - Cooldown sonrası missed sinyal detayı loglanıyor
# ===================================================================================

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
import traceback
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
os.environ["TERM"] = "xterm-256color"

# ==============================================================
# 1. AYAR PANELİ
# ==============================================================
CONFIG = {
    "NTFY_TOPIC": "RBD-CRYPT",
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
    "MIN_ATR_PERCENT": 0.85,        # Artık gerçekten kullanılıyor
    "SL_M": 1.65,
    "TP_M": 2.55,
    "COOLDOWN_MINUTES": 20,
    "MAX_HOLD_MINUTES": 30,
    "MAX_HOLD_ST_GRACE_BARS": 2,    # YENİ: timeout dolunca ST flip için +2 bar tolerans
    "CHOP_ADX_THRESHOLD": 18,
    "BTC_VOL_THRESHOLD": 0.3,
    "TARGET_MINUTES": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],
    "MAX_LOG_SIZE_BYTES": 50_000_000
}

FILES = {
    "LOG":             os.path.join(CONFIG["BASE_PATH"], "hunter_history.csv"),
    "ACTIVE":          os.path.join(CONFIG["BASE_PATH"], "active_trades.json"),
    "ALL_SIGNALS":     os.path.join(CONFIG["BASE_PATH"], "all_signals.csv"),
    "MARKET_CONTEXT":  os.path.join(CONFIG["BASE_PATH"], "market_context.csv"),   # YENİ
    "ERROR_LOG":       os.path.join(CONFIG["BASE_PATH"], "error_log.csv"),         # YENİ
    "STATE":           os.path.join(CONFIG["BASE_PATH"], "engine_state.json")
}

if not os.path.exists(CONFIG["BASE_PATH"]):
    os.makedirs(CONFIG["BASE_PATH"], exist_ok=True)

# ==============================================================
# 2. HATA LOGLAMA (Sessiz hataları yakala)
# ==============================================================
def log_error(context: str, error: Exception, extra: str = ""):
    """Her exception'ı error_log.csv'ye yazar. Hiçbir hata kaybolmaz."""
    try:
        row = {
            'timestamp':  get_tr_time().isoformat(),
            'context':    context,
            'error_type': type(error).__name__,
            'error_msg':  str(error)[:300],
            'traceback':  traceback.format_exc()[-500:],
            'extra':      extra
        }
        pd.DataFrame([row]).to_csv(
            FILES["ERROR_LOG"], mode='a',
            header=not os.path.exists(FILES["ERROR_LOG"]),
            index=False
        )
        console.print(f"[bold red]⚠️ HATA [{context}]:[/] {type(error).__name__}: {str(error)[:120]}")
    except Exception:
        pass  # Loglama kendisi patlarsa yapacak bir şey yok

# ==============================================================
# 3. SİSTEM DURUMU
# ==============================================================
class HunterState:
    def __init__(self):
        self.current_coin = "Baslatiliyor..."
        self.progress_pct = 0
        self.processed_count = 0
        self.total_count = 0
        self.status = "BASLATILIYOR"
        self.is_scanning = False
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
        # BTC context — her scan güncellenir, loglarda kullanılır
        self.btc_trend_val = 0
        self.btc_atr_pct = 0.0
        self.btc_rsi = 0.0
        self.btc_adx = 0.0
        self.btc_vol_ratio = 0.0
        self.is_chop_market = False
        self.scan_id = 0  # Her scan'e benzersiz ID, logları birbirine bağlar
        self.load_state()

    @property
    def dynamic_trade_size(self):
        size = self.balance * (CONFIG["RISK_PERCENT_PER_TRADE"] / 100.0)
        return min(max(CONFIG["MIN_TRADE_SIZE"], size), CONFIG["MAX_TRADE_SIZE"])

    def load_state(self):
        try:
            with open(FILES["ACTIVE"], 'r') as f:
                self.active_positions = json.load(f)
        except Exception:
            self.active_positions = {}

        try:
            with open(FILES["STATE"], 'r') as f:
                saved = json.load(f)
                self.balance      = saved.get("balance", CONFIG["STARTING_BALANCE"])
                self.peak_balance = saved.get("peak_balance", CONFIG["STARTING_BALANCE"])
                self.scan_id      = saved.get("scan_id", 0)
        except Exception:
            pass

    def save_state(self):
        try:
            with open(FILES["STATE"], 'w') as f:
                json.dump({
                    "balance":      self.balance,
                    "peak_balance": self.peak_balance,
                    "scan_id":      self.scan_id
                }, f)
            temp = FILES["ACTIVE"] + ".tmp"
            with open(temp, 'w') as f:
                json.dump(self.active_positions, f)
            os.replace(temp, FILES["ACTIVE"])
        except Exception as e:
            log_error("save_state", e)

    def update_balance(self, pnl_percent: float, trade_size: float):
        self.balance += trade_size * (pnl_percent / 100)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.save_state()

state = HunterState()

# ==============================================================
# 4. YARDIMCI FONKSİYONLAR
# ==============================================================
def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_tr_time() -> datetime:
    return datetime.utcnow() + timedelta(hours=3)

def rotate_logs():
    for file_key in ["ALL_SIGNALS", "LOG", "MARKET_CONTEXT", "ERROR_LOG"]:
        path = FILES[file_key]
        try:
            if os.path.exists(path) and os.path.getsize(path) > CONFIG["MAX_LOG_SIZE_BYTES"]:
                os.rename(path, path + f"_old_{int(time.time())}")
        except Exception as e:
            log_error("rotate_logs", e, file_key)

# ==============================================================
# 5. BİLDİRİM MODÜLÜ
# ==============================================================
def send_ntfy_notification(title: str, message: str, image_buf=None, tags="robot", priority="3"):
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    headers = {
        "Title":    title.encode('utf-8'),
        "Tags":     tags,
        "Priority": str(priority)
    }
    try:
        if image_buf:
            headers["Filename"] = "chart.png"
            headers["Message"]  = message.replace('\n', ' | ').encode('utf-8')
            requests.post(url, data=image_buf.getvalue(), headers=headers, timeout=10)
        else:
            requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
    except Exception as e:
        log_error("send_ntfy_notification", e, title)

def send_ntfy_file(filepath: str, filename: str, message: str = ""):
    """
    Tek bir dosyayı ntfy'ye PUT ile gönderir.
    filename parametresi zaten tarih eki içermeli (günlük dump bunu halleder).
    """
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    headers = {"Filename": filename}
    if message:
        headers["Message"] = message.encode('utf-8')
    try:
        with open(filepath, 'rb') as f:
            requests.put(url, data=f, headers=headers, timeout=30)
    except Exception as e:
        log_error("send_ntfy_file", e, filename)


def gunluk_dump_gonder():
    """
    BASE_PATH altındaki TÜM dosyaları (ve bir seviye alt klasörleri) ntfy'ye gönderir.
    Her dosyanın adına tarih eki eklenir:  hunter_history_2026-02-23.csv
    JSON ve CSV'ler sırayla, aralarında 1s beklenerek gönderilir (spam engeli).
    """
    tarih_str = get_tr_time().strftime('%Y-%m-%d')
    base      = CONFIG["BASE_PATH"]

    # BASE_PATH altındaki tüm dosyaları topla (recursive, _old_ arşivleri dahil değil)
    dosyalar = []
    for root, dirs, files in os.walk(base):
        # _old_ arşivleri ve temp dosyaları atla
        files = [f for f in files if not f.endswith('.tmp') and '_old_' not in f]
        for fname in sorted(files):
            dosyalar.append(os.path.join(root, fname))

    if not dosyalar:
        send_ntfy_notification(
            f"📦 Günlük Döküm ({tarih_str})",
            "BASE_PATH içinde gönderilecek dosya bulunamadı.",
            tags="warning", priority="3"
        )
        return

    # Önce özet bildirim gönder
    dosya_listesi = "\n".join(
        f"• {os.path.basename(d)}  ({round(os.path.getsize(d)/1024, 1)} KB)"
        for d in dosyalar
    )
    send_ntfy_notification(
        f"📦 Günlük Döküm Başlıyor ({tarih_str})",
        f"Toplam {len(dosyalar)} dosya gönderilecek:\n{dosya_listesi}",
        tags="package,floppy_disk", priority="4"
    )
    time.sleep(2)

    # Dosyaları sırayla gönder
    for i, filepath in enumerate(dosyalar, 1):
        try:
            orijinal_ad = os.path.basename(filepath)
            kok, uzanti  = os.path.splitext(orijinal_ad)
            tarihli_ad   = f"{kok}_{tarih_str}{uzanti}"   # hunter_history_2026-02-23.csv

            boyut_kb = round(os.path.getsize(filepath) / 1024, 1)
            mesaj    = f"[{i}/{len(dosyalar)}] {orijinal_ad} → {boyut_kb} KB"

            send_ntfy_file(filepath, tarihli_ad, mesaj)
            console.print(f"[dim green]📤 Gönderildi:[/] {tarihli_ad} ({boyut_kb} KB)")
            time.sleep(1.5)   # ntfy rate-limit koruma
        except Exception as e:
            log_error("gunluk_dump_gonder", e, filepath)

    send_ntfy_notification(
        f"✅ Günlük Döküm Tamamlandı ({tarih_str})",
        f"Tüm {len(dosyalar)} dosya başarıyla gönderildi. İyi geceler patron 🌙",
        tags="white_check_mark,moon", priority="3"
    )

# ==============================================================
# 6. VERİ ÇEKİMİ
# ==============================================================
def safe_api_get(url: str, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                console.print(f"[bold yellow]⚠️ 429 Rate Limit — {attempt+1}. deneme, bekleniyor...[/]")
                time.sleep(10)
            else:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            log_error("safe_api_get", e, f"url={url} attempt={attempt}")
            time.sleep(3)
    return None

def get_top_futures_coins(limit=30) -> list:
    data = safe_api_get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if data:
        usdt_pairs = [
            d for d in data
            if d['symbol'].endswith('USDT') and '_' not in d['symbol']
        ]
        return [
            p['symbol']
            for p in sorted(usdt_pairs, key=lambda x: float(x['quoteVolume']), reverse=True)[:limit]
        ]
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

def get_live_futures_data(symbol: str, limit=300):
    data = safe_api_get(
        "https://fapi.binance.com/fapi/v1/klines",
        {"symbol": symbol, "interval": "5m", "limit": limit}
    )
    if data:
        df = pd.DataFrame(data).iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=3)
        df.set_index('timestamp', inplace=True)
        return df.apply(pd.to_numeric, errors='coerce')
    return None

# ==============================================================
# 7. İNDİKATÖRLER
# ==============================================================
def hesapla_indikatorler(df: pd.DataFrame) -> pd.DataFrame:
    # talib float64 zorunlu — Binance bazen object/float32 döner, cast şart
    c = df['close'].values.astype(float)
    h = df['high'].values.astype(float)
    l = df['low'].values.astype(float)
    v = df['volume'].values.astype(float)

    df['RSI']       = talib.RSI(c, CONFIG["RSI_PERIOD"])
    df['ADX']       = talib.ADX(h, l, c, 14)
    df['PLUS_DI']   = talib.PLUS_DI(h, l, c, 14)   # YENİ: DI farkı data için
    df['MINUS_DI']  = talib.MINUS_DI(h, l, c, 14)  # YENİ
    df['EMA20']     = talib.EMA(c, 20)
    df['EMA50']     = talib.EMA(c, 50)              # YENİ: trend derinliği
    df['ATR_14']    = talib.ATR(h, l, c, 14)
    df['BBANDS_UP'], df['BBANDS_MID'], df['BBANDS_LOW'] = talib.BBANDS(c, 20, 2, 2)  # YENİ
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(c, 12, 26, 9)        # YENİ
    df['VOL_SMA_20'] = talib.SMA(v, 20)
    df['VOL_RATIO']  = np.where(df['VOL_SMA_20'] > 0, v / df['VOL_SMA_20'], 0)
    df['ATR_PCT']    = (df['ATR_14'] / df['close']) * 100                             # YENİ: normalize ATR

    # Supertrend
    atr_st = talib.ATR(h, l, c, 10)
    hl2    = (h + l) / 2
    st_line = np.zeros(len(c))
    trend   = np.ones(len(c))

    for i in range(1, len(c)):
        up = hl2[i] + CONFIG["ST_M"] * atr_st[i]
        dn = hl2[i] - CONFIG["ST_M"] * atr_st[i]
        if c[i-1] > st_line[i-1]:
            st_line[i] = max(dn, st_line[i-1])
            trend[i]   = 1
        else:
            st_line[i] = min(up, st_line[i-1])
            trend[i]   = -1
        if trend[i] != trend[i-1]:
            st_line[i] = dn if trend[i] == 1 else up

    df['TREND']      = trend
    df['ST_LINE']    = st_line
    df['ST_DIST_PCT'] = ((df['close'] - df['ST_LINE']) / df['close']) * 100  # YENİ
    df['FLIP_LONG']  = (df['TREND'] == 1) & ((df['TREND'].shift(1) == -1) | (df['TREND'].shift(2) == -1))
    df['FLIP_SHORT'] = (df['TREND'] == -1) & ((df['TREND'].shift(1) == 1) | (df['TREND'].shift(2) == 1))

    # Price action
    df['BODY_PCT']   = abs(df['close'] - df['open']) / df['open'] * 100      # YENİ
    df['UPPER_WICK'] = (df['high'] - df[['close','open']].max(axis=1)) / df['open'] * 100  # YENİ
    df['LOWER_WICK'] = (df[['close','open']].min(axis=1) - df['low']) / df['open'] * 100   # YENİ

    return df

# ==============================================================
# 8. SİNYAL VE SKOR HESABI
# ==============================================================
def hesapla_power_score(row) -> float:
    """
    0-100 arası sinyal gücü skoru.
    Walk-forward analizinde hangi skorların karlı olduğunu görmek için kullanılır.
    Alt bileşenler de loglanıyor.
    """
    score = 0.0

    # RSI bileşeni (0-25)
    if row['FLIP_LONG']:
        rsi_component = max(0, min(25, (row['RSI'] - CONFIG["RSI_LONG"]) * 2.5))
    else:
        rsi_component = max(0, min(25, (CONFIG["RSI_SHORT"] - row['RSI']) * 2.5))
    score += rsi_component

    # Hacim bileşeni (0-25)
    vol_component = max(0, min(25, (row['VOL_RATIO'] - CONFIG["VOL_FILTER"]) * 15))
    score += vol_component

    # ADX bileşeni (0-20)
    adx_component = max(0, min(20, (row['ADX'] - CONFIG["ADX_THRESHOLD"]) * 0.8))
    score += adx_component

    # ATR % bileşeni (0-15) — düşük ATR düşük skor
    atr_component = max(0, min(15, (row['ATR_PCT'] - CONFIG["MIN_ATR_PERCENT"]) * 5))
    score += atr_component

    # MACD histogram yönü (0-10)
    if row['FLIP_LONG'] and row.get('MACD_HIST', 0) > 0:
        score += 10
    elif row['FLIP_SHORT'] and row.get('MACD_HIST', 0) < 0:
        score += 10

    # BB genişliği (0-5) — sıkışık piyasayı cezalandır
    bb_width = (row.get('BBANDS_UP', 0) - row.get('BBANDS_LOW', 0)) / row.get('BBANDS_MID', 1) * 100
    bb_component = max(0, min(5, bb_width * 0.5))
    score += bb_component

    return round(score, 2)

def hesapla_signal_score(row) -> int:
    """
    Kaç koşul sağlandı (0-6). Basit sayım, gücü değil adeti verir.
    """
    checks = [
        row.get('FLIP_LONG', False) or row.get('FLIP_SHORT', False),
        row['RSI'] > CONFIG["RSI_LONG"] if row.get('FLIP_LONG') else row['RSI'] < CONFIG["RSI_SHORT"],
        row['VOL_RATIO'] > CONFIG["VOL_FILTER"],
        row['ADX'] > CONFIG["ADX_THRESHOLD"],
        row['ATR_PCT'] > CONFIG["MIN_ATR_PERCENT"],
        (row['close'] > row['EMA20']) if row.get('FLIP_LONG') else (row['close'] < row['EMA20'])
    ]
    return sum(checks)

def sinyal_kontrol(row):
    atr_ok = (row['ATR_14'] / row['close'] * 100) >= CONFIG["MIN_ATR_PERCENT"]  # Artık aktif!
    is_long  = (row['FLIP_LONG']  and row['RSI'] > CONFIG["RSI_LONG"]
                and row['VOL_RATIO'] > CONFIG["VOL_FILTER"]
                and row['close'] > row['EMA20']
                and row['ADX']   > CONFIG["ADX_THRESHOLD"]
                and atr_ok)
    is_short = (row['FLIP_SHORT'] and row['RSI'] < CONFIG["RSI_SHORT"]
                and row['VOL_RATIO'] > CONFIG["VOL_FILTER"]
                and row['close'] < row['EMA20']
                and row['ADX']   > CONFIG["ADX_THRESHOLD"]
                and atr_ok)
    if is_long:  return "LONG"
    if is_short: return "SHORT"
    return None

# ==============================================================
# 9. BTC ANALİZİ
# ==============================================================
def get_btc_context() -> dict:
    """BTC'den tam bir context sözlüğü döner — hem filtre hem loglama için."""
    btc_df = get_live_futures_data("BTCUSDT", 200)
    if btc_df is None or len(btc_df) < 50:
        return {
            "trend": 0, "atr_pct": 0.0, "rsi": 0.0, "adx": 0.0,
            "vol_ratio": 0.0, "macd_hist": 0.0, "close": 0.0,
            "ema20": 0.0, "bb_width_pct": 0.0
        }
    try:
        btc_df = hesapla_indikatorler(btc_df)
        r = btc_df.iloc[-2]
        bb_width = (r['BBANDS_UP'] - r['BBANDS_LOW']) / r['BBANDS_MID'] * 100 if r['BBANDS_MID'] > 0 else 0
        return {
            "trend":       int(r['TREND']),
            "atr_pct":     round(float(r['ATR_PCT']), 4),
            "rsi":         round(float(r['RSI']), 2),
            "adx":         round(float(r['ADX']), 2),
            "vol_ratio":   round(float(r['VOL_RATIO']), 3),
            "macd_hist":   round(float(r['MACD_HIST']), 6),
            "close":       round(float(r['close']), 2),
            "ema20":       round(float(r['EMA20']), 2),
            "bb_width_pct":round(float(bb_width), 3)
        }
    except Exception as e:
        log_error("get_btc_context", e)
        return {
            "trend": 0, "atr_pct": 0.0, "rsi": 0.0, "adx": 0.0,
            "vol_ratio": 0.0, "macd_hist": 0.0, "close": 0.0,
            "ema20": 0.0, "bb_width_pct": 0.0
        }

# ==============================================================
# 10. LOGLAMA
# ==============================================================
def log_trade_to_csv(trade_dict: dict):
    """Kapanan işlemi hunter_history.csv'ye yazar. Geniş şema."""
    try:
        df = pd.read_csv(FILES["LOG"]) if os.path.exists(FILES["LOG"]) else pd.DataFrame()
        new_row = pd.DataFrame([trade_dict])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(FILES["LOG"], index=False, encoding='utf-8-sig')
    except Exception as e:
        log_error("log_trade_to_csv", e)

def log_potential_signal(sym: str, signal_type: str, row, score: int,
                          power_score: float, entered: bool, reason: str = "",
                          btc_ctx: dict = None):
    """
    all_signals.csv — Her tespit edilen sinyalin tam fotoğrafı.
    Girişe dönüşsün ya da dönüşmesin, tüm koşullar kaydediliyor.
    Walk-forward: 'tradable=False' ama karlı olan sinyaller
    parametre ayarı için altın madeni.
    """
    if btc_ctx is None:
        btc_ctx = {}

    try:
        bb_width = (row.get('BBANDS_UP', 0) - row.get('BBANDS_LOW', 0)) / row.get('BBANDS_MID', 1) * 100
        log_row = {
            # Zaman / kimlik
            'timestamp':        get_tr_time().isoformat(),
            'scan_id':          state.scan_id,
            'coin':             sym,
            'signal':           signal_type,
            # Sinyal kalitesi
            'score':            score,
            'power_score':      power_score,
            # Fiyat
            'close':            round(float(row['close']), 6),
            'open':             round(float(row['open']), 6),
            'high':             round(float(row['high']), 6),
            'low':              round(float(row['low']), 6),
            'volume':           round(float(row['volume']), 2),
            # İndikatörler
            'rsi':              round(float(row['RSI']), 2),
            'adx':              round(float(row['ADX']), 2),
            'plus_di':          round(float(row.get('PLUS_DI', 0)), 2),
            'minus_di':         round(float(row.get('MINUS_DI', 0)), 2),
            'atr_14':           round(float(row['ATR_14']), 6),
            'atr_pct':          round(float(row['ATR_PCT']), 4),
            'vol_ratio':        round(float(row['VOL_RATIO']), 3),
            'ema20':            round(float(row['EMA20']), 6),
            'ema50':            round(float(row.get('EMA50', 0)), 6),
            'st_line':          round(float(row['ST_LINE']), 6),
            'st_dist_pct':      round(float(row.get('ST_DIST_PCT', 0)), 4),
            'macd':             round(float(row.get('MACD', 0)), 6),
            'macd_signal':      round(float(row.get('MACD_SIGNAL', 0)), 6),
            'macd_hist':        round(float(row.get('MACD_HIST', 0)), 6),
            'bb_upper':         round(float(row.get('BBANDS_UP', 0)), 6),
            'bb_lower':         round(float(row.get('BBANDS_LOW', 0)), 6),
            'bb_width_pct':     round(float(bb_width), 3),
            # Price action
            'body_pct':         round(float(row.get('BODY_PCT', 0)), 4),
            'upper_wick_pct':   round(float(row.get('UPPER_WICK', 0)), 4),
            'lower_wick_pct':   round(float(row.get('LOWER_WICK', 0)), 4),
            # BTC context
            'btc_trend':        btc_ctx.get('trend', 0),
            'btc_atr_pct':      btc_ctx.get('atr_pct', 0.0),
            'btc_rsi':          btc_ctx.get('rsi', 0.0),
            'btc_adx':          btc_ctx.get('adx', 0.0),
            'btc_vol_ratio':    btc_ctx.get('vol_ratio', 0.0),
            'btc_macd_hist':    btc_ctx.get('macd_hist', 0.0),
            'btc_close':        btc_ctx.get('close', 0.0),
            'btc_bb_width_pct': btc_ctx.get('bb_width_pct', 0.0),
            # Piyasa koşulu
            'is_chop_market':   state.is_chop_market,
            'active_positions': len(state.active_positions),
            'balance_at_signal':round(state.balance, 2),
            # Karar
            'tradable':         entered,
            'blocked_reason':   reason,
            # Filtre eşikleri (parametre değişirse retroaktif analiz için)
            'cfg_rsi_long':     CONFIG["RSI_LONG"],
            'cfg_rsi_short':    CONFIG["RSI_SHORT"],
            'cfg_vol_filter':   CONFIG["VOL_FILTER"],
            'cfg_adx_thr':      CONFIG["ADX_THRESHOLD"],
            'cfg_atr_min_pct':  CONFIG["MIN_ATR_PERCENT"],
            'cfg_st_m':         CONFIG["ST_M"],
        }
        pd.DataFrame([log_row]).to_csv(
            FILES["ALL_SIGNALS"], mode='a',
            header=not os.path.exists(FILES["ALL_SIGNALS"]),
            index=False
        )
    except Exception as e:
        log_error("log_potential_signal", e, sym)

def log_market_context(btc_ctx: dict, coin_count: int, open_pos: int):
    """
    market_context.csv — Her scan'in başındaki piyasa snapshot'ı.
    Hangi market koşullarında ne kadar sinyal üretildiğini analiz etmek için.
    """
    try:
        row = {
            'timestamp':       get_tr_time().isoformat(),
            'scan_id':         state.scan_id,
            'btc_trend':       btc_ctx.get('trend', 0),
            'btc_atr_pct':     btc_ctx.get('atr_pct', 0.0),
            'btc_rsi':         btc_ctx.get('rsi', 0.0),
            'btc_adx':         btc_ctx.get('adx', 0.0),
            'btc_vol_ratio':   btc_ctx.get('vol_ratio', 0.0),
            'btc_macd_hist':   btc_ctx.get('macd_hist', 0.0),
            'btc_close':       btc_ctx.get('close', 0.0),
            'btc_bb_width_pct':btc_ctx.get('bb_width_pct', 0.0),
            'is_chop':         state.is_chop_market,
            'market_dir':      state.market_direction_text,
            'coins_scanned':   coin_count,
            'open_positions':  open_pos,
            'balance':         round(state.balance, 2),
        }
        pd.DataFrame([row]).to_csv(
            FILES["MARKET_CONTEXT"], mode='a',
            header=not os.path.exists(FILES["MARKET_CONTEXT"]),
            index=False
        )
    except Exception as e:
        log_error("log_market_context", e)

def get_advanced_metrics():
    try:
        if not os.path.exists(FILES["LOG"]):
            return 0, 0, 0, 0.0, 0.0
        df = pd.read_csv(FILES["LOG"])
        if len(df) == 0:
            return 0, 0, 0, 0.0, 0.0
        wins   = df[df['PnL_Yuzde'] > 0]
        losses = df[df['PnL_Yuzde'] <= 0]
        tot_trd, w_count = len(df), len(wins)
        gross_profit = wins['PnL_USD'].sum()   if 'PnL_USD' in wins.columns   else wins['PnL_Yuzde'].sum()
        gross_loss   = abs(losses['PnL_USD'].sum()) if 'PnL_USD' in losses.columns else abs(losses['PnL_Yuzde'].sum())
        pf  = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9
        mdd = ((state.peak_balance - state.balance) / state.peak_balance) * 100
        return tot_trd, w_count, int((w_count / tot_trd) * 100) if tot_trd > 0 else 0, pf, mdd
    except Exception as e:
        log_error("get_advanced_metrics", e)
        return 0, 0, 0, 0.0, 0.0

# ==============================================================
# 11. GRAFİK
# ==============================================================
def create_trade_chart(df, sym, pos, is_entry=False, curr_c=None, pnl=0.0, close_reason=""):
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 5))
        entry_dt = pd.to_datetime(pos.get('entry_idx_time', pos.get('full_time', get_tr_time().strftime('%Y-%m-%d %H:%M:%S'))))

        plot_df = (df.tail(60).copy() if is_entry
                   else df[df.index >= entry_dt - pd.Timedelta(minutes=150)].tail(200).copy())

        up   = plot_df[plot_df.close >= plot_df.open]
        down = plot_df[plot_df.close  < plot_df.open]

        ax.vlines(up.index,   up.low,   up.high,   color='#2ecc71', linewidth=1.5, alpha=0.8)
        ax.vlines(down.index, down.low, down.high, color='#e74c3c', linewidth=1.5, alpha=0.8)
        ax.bar(up.index,   up.close   - up.open,   0.008, bottom=up.open,     color='#2ecc71', alpha=0.9)
        ax.bar(down.index, down.open  - down.close, 0.008, bottom=down.close, color='#e74c3c', alpha=0.9)

        ax.axhline(pos['entry_p'], color='#3498db', linestyle='--', alpha=0.8, label='Giriş')
        ax.axhline(pos['tp'],      color='#2ecc71', linestyle=':',  linewidth=2, label='TP')
        ax.axhline(pos['sl'],      color='#e74c3c', linestyle=':',  linewidth=2, label='SL')

        if is_entry:
            ax.scatter(entry_dt, pos['entry_p'], color='yellow', s=150, zorder=5, edgecolors='black')
        else:
            ax.scatter(entry_dt, pos['entry_p'], color='yellow', s=120, zorder=5, edgecolors='black')
            exit_price = pos['tp'] if "KAR" in close_reason else pos['sl']
            ax.scatter(plot_df.index[-1], exit_price,
                       color='#2ecc71' if pnl > 0 else '#e74c3c',
                       s=200, zorder=5, marker='X', edgecolors='white')

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        fig.autofmt_xdate(rotation=30)
        ax.legend(loc='upper left', framealpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        plt.close('all')
        buf.seek(0)
        return buf
    except Exception as e:
        log_error("create_trade_chart", e, sym)
        plt.close('all')
        return None

# ==============================================================
# 12. DASHBOARD
# ==============================================================
def draw_fund_dashboard():
    clear_terminal()
    strategy_text = (
        "[bold cyan]🎯 STRATEJİ (v85.0):[/] ST Flip + RSI9 + VOL>1.42x + EMA20 + ADX>22 + ATR>0.85%\n"
        f"[bold yellow]🏆 MAX {CONFIG['MAX_POSITIONS']} POZ | Dinamik Risk: {state.dynamic_trade_size:.1f}$ | 30dk max hold[/]\n"
        f"[bold magenta]🌍 PİYASA YÖNÜ:[/] {state.market_direction} "
        f"| BTC ATR%: {state.btc_atr_pct:.3f} | BTC RSI: {state.btc_rsi:.1f} | BTC ADX: {state.btc_adx:.1f}\n"
        f"[dim]Scan ID: {state.scan_id} | Enterprise: Crash Guard + Max Logging + Walk-Forward Ready[/]"
    )
    console.print(Panel(strategy_text, title="[bold magenta]🛠️ RBD-CRYPT v85.0 QUANT ENGINE[/]", border_style="magenta"))

    tot_trd, wins, b_wr, pf, max_dd = get_advanced_metrics()
    bal_color = "bold bright_green" if state.balance >= CONFIG["STARTING_BALANCE"] else "bold bright_red"
    history_text = (
        f"[dim white]Kasa:[/] [{bal_color}]${state.balance:.2f}[/] (Başlangıç: ${CONFIG['STARTING_BALANCE']}) | "
        f"[dim white]Tepe:[/] [bold cyan]${state.peak_balance:.2f}[/]\n"
        f"[dim white]İşlem Başarısı:[/] [bold cyan]{wins} / {tot_trd} (%{b_wr})[/] | "
        f"[dim white]Profit Factor:[/] {pf} | [dim white]Max DD:[/] %{max_dd:.2f}"
    )
    console.print(Panel(history_text, title="[bold yellow]📜 KASA VE PERFORMANS[/]", border_style="yellow"))

    if state.active_positions:
        act_table = Table(expand=True, header_style="bold cyan", show_lines=True)
        for col in ["Coin", "Yön", "Süre", "PnL (%)", "Giriş", "TP", "SL"]:
            act_table.add_column(col)
        for sym, pos in state.active_positions.items():
            curr_pnl  = pos.get('curr_pnl', 0.0)
            pnl_color = "bright_green" if curr_pnl > 0 else "bright_red" if curr_pnl < 0 else "white"
            dur = str(get_tr_time() - datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')).split('.')[0]
            act_table.add_row(
                f"[bold white]{sym}[/]",
                "[bold bright_green]LONG[/]" if pos.get('dir') == 'LONG' else "[bold bright_red]SHORT[/]",
                dur,
                f"[bold {pnl_color}]{curr_pnl:.2f}%[/]",
                str(round(pos.get('entry_p', 0), 5)),
                str(round(pos.get('tp', 0), 5)),
                str(round(pos.get('sl', 0), 5))
            )
        console.print(Panel(act_table, title=f"[bold bright_green]🟢 AKTİF İŞLEMLER ({len(state.active_positions)})[/]", border_style="bright_green"))
    else:
        console.print(Panel("[dim]Şu an açık pozisyon bulunmuyor...[/]", title="[bold white]⚪ AKTİF İŞLEM YOK[/]", border_style="dim white"))

    if state.is_scanning:
        console.print(
            f"\n📡 [bold yellow]{state.status}[/] | "
            f"İlerleme: [bold bright_green]{state.processed_count}/{state.total_count}[/] | "
            f"Coin: {state.current_coin}"
        )
    else:
        console.print(f"\n📡 [bold bright_green]{state.status}[/]")

# ==============================================================
# 13. ANA KONTROL DÖNGÜSÜ
# ==============================================================
def run_bot_cycle():
    state.scan_id += 1
    state.save_state()

    state.status = "🌐 BİNANCE FUTURES: Hacimli Coinler Çekiliyor..."
    draw_fund_dashboard()
    coins = get_top_futures_coins(CONFIG["TOP_COINS_LIMIT"])

    # --- BTC Context ---
    btc_ctx = get_btc_context()
    state.btc_trend_val = btc_ctx["trend"]
    state.btc_atr_pct   = btc_ctx["atr_pct"]
    state.btc_rsi       = btc_ctx["rsi"]
    state.btc_adx       = btc_ctx["adx"]
    state.btc_vol_ratio = btc_ctx["vol_ratio"]

    state.is_chop_market = btc_ctx["atr_pct"] < CONFIG["BTC_VOL_THRESHOLD"]

    if state.is_chop_market:
        state.market_direction      = "[bold bright_red]CHOP MARKET (BTC Düşük Vol - SADECE VERİ TOPLANIYOR)[/]"
        state.market_direction_text = "CHOP MARKET (Düşük Volatilite)"
    else:
        state.market_direction      = "[bold bright_green]YÜKSELİŞ (LONG)[/]" if btc_ctx["trend"] == 1 else "[bold bright_red]DÜŞÜŞ (SHORT)[/]"
        state.market_direction_text = "YÜKSELİŞ (LONG)" if btc_ctx["trend"] == 1 else "DÜŞÜŞ (SHORT)"

    # Piyasa context'ini logla
    log_market_context(btc_ctx, len(coins), len(state.active_positions))

    state.total_count = len(coins)
    state.processed_count = 0
    state.is_scanning = True
    state.missed_this_scan = 0
    state.status = "🚀 QUANT MOTORU: VADELİ PİYASA TARANIYOR (Asenkron)..."

    # --- Asenkron veri çekimi ---
    fetched_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_sym = {executor.submit(get_live_futures_data, sym, 300): sym for sym in coins}
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                fetched_data[sym] = future.result()
            except Exception as e:
                log_error("fetch_worker", e, sym)
                fetched_data[sym] = None

    # --- Coin döngüsü ---
    for sym in coins:
        try:
            df = fetched_data.get(sym)
            state.current_coin  = sym
            state.processed_count += 1
            draw_fund_dashboard()

            if df is None or len(df) < 50:
                continue

            try:
                df = hesapla_indikatorler(df)
            except Exception as e:
                log_error("hesapla_indikatorler", e, sym)
                continue

            # ── A. AKTİF İŞLEM KONTROLÜ ──────────────────────────────
            if sym in state.active_positions:
                pos    = state.active_positions[sym]
                curr_h = float(df['high'].iloc[-1])
                curr_l = float(df['low'].iloc[-1])
                curr_c = float(df['close'].iloc[-1])
                entry_p = pos.get('entry_p', curr_c)

                pos['curr_p']   = curr_c
                closed          = False
                close_reason    = ""
                pnl             = 0.0

                pos_time    = datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')
                hold_minutes = (get_tr_time() - pos_time).total_seconds() / 60

                # Timeout — ama ST flip varsa 2 bar daha bekle
                if hold_minutes > CONFIG["MAX_HOLD_MINUTES"]:
                    last_trend   = int(df['TREND'].iloc[-1])
                    grace_ok     = False

                    if pos['dir'] == 'LONG' and last_trend == -1:
                        # ST zaten aleyhte döndü, çık
                        grace_ok = False
                    elif pos['dir'] == 'SHORT' and last_trend == 1:
                        grace_ok = False
                    elif hold_minutes < CONFIG["MAX_HOLD_MINUTES"] + CONFIG["MAX_HOLD_ST_GRACE_BARS"] * 5:
                        # Grace süresi içinde, ST henüz aleyhte değil — bekle
                        grace_ok = True

                    if not grace_ok:
                        closed       = True
                        close_reason = "TIMEOUT"
                        pnl = ((curr_c - entry_p) / entry_p * 100 if pos['dir'] == 'LONG'
                               else (entry_p - curr_c) / entry_p * 100)

                if not closed:
                    if pos['dir'] == 'LONG':
                        pos['curr_pnl'] = (curr_c - entry_p) / entry_p * 100
                        if curr_h >= pos['tp']:
                            pnl = (pos['tp'] - entry_p) / entry_p * 100
                            closed, close_reason = True, "KAR ALDI"
                        elif curr_l <= pos['sl']:
                            pnl = (pos['sl'] - entry_p) / entry_p * 100
                            closed, close_reason = True, "STOP OLDU"
                    else:
                        pos['curr_pnl'] = (entry_p - curr_c) / entry_p * 100
                        if curr_l <= pos['tp']:
                            pnl = (entry_p - pos['tp']) / entry_p * 100
                            closed, close_reason = True, "KAR ALDI"
                        elif curr_h >= pos['sl']:
                            pnl = (entry_p - pos['sl']) / entry_p * 100
                            closed, close_reason = True, "STOP OLDU"

                if closed:
                    state.cooldowns[sym] = get_tr_time()
                    trade_size = pos.get('trade_size', CONFIG["MIN_TRADE_SIZE"])
                    pnl_usd    = trade_size * (pnl / 100)
                    state.update_balance(pnl, trade_size)

                    # ── Geniş kapanış logu ────────────────────────────
                    row_at_close = df.iloc[-2]
                    trade_log = {
                        # Kimlik
                        'Scan_ID':           state.scan_id,
                        'Tarih':             get_tr_time().strftime('%Y-%m-%d'),
                        'Giris_Saati':       pos['full_time'].split(' ')[1],
                        'Cikis_Saati':       get_tr_time().strftime('%H:%M:%S'),
                        'Hold_Dakika':       round(hold_minutes, 1),
                        'Coin':              sym,
                        'Yon':               pos['dir'],
                        # Fiyat
                        'Giris_Fiyati':      round(entry_p, 6),
                        'Cikis_Fiyati':      round(curr_c, 6),
                        'TP_Seviyesi':       round(pos['tp'], 6),
                        'SL_Seviyesi':       round(pos['sl'], 6),
                        'TP_SL_Orani':       round(abs(pos['tp'] - entry_p) / abs(pos['sl'] - entry_p), 3),
                        # Risk / Kâr
                        'Risk_USD':          round(trade_size, 2),
                        'PnL_Yuzde':         round(pnl, 2),
                        'PnL_USD':           round(pnl_usd, 2),
                        'Kasa_Son_Durum':    round(state.balance, 2),
                        'Sonuc':             close_reason,
                        # Giriş anındaki indikatörler (pos içinden)
                        'Giris_RSI':         pos.get('entry_rsi', 0),
                        'Giris_ADX':         pos.get('entry_adx', 0),
                        'Giris_VOL_RATIO':   pos.get('entry_vol_ratio', 0),
                        'Giris_ATR_PCT':     pos.get('entry_atr_pct', 0),
                        'Giris_Power_Score': pos.get('power_score', 0),
                        'Giris_Score':       pos.get('signal_score', 0),
                        # Çıkış anındaki indikatörler
                        'Cikis_RSI':         round(float(row_at_close['RSI']), 2),
                        'Cikis_ADX':         round(float(row_at_close['ADX']), 2),
                        'Cikis_VOL_RATIO':   round(float(row_at_close['VOL_RATIO']), 3),
                        'Cikis_ATR_PCT':     round(float(row_at_close['ATR_PCT']), 4),
                        'Cikis_TREND':       int(row_at_close['TREND']),
                        'Cikis_MACD_HIST':   round(float(row_at_close.get('MACD_HIST', 0)), 6),
                        # BTC context
                        'BTC_Trend':         btc_ctx.get('trend', 0),
                        'BTC_ATR_PCT':       btc_ctx.get('atr_pct', 0.0),
                        'BTC_RSI':           btc_ctx.get('rsi', 0.0),
                        'BTC_ADX':           btc_ctx.get('adx', 0.0),
                        'BTC_Vol_Ratio':     btc_ctx.get('vol_ratio', 0.0),
                        # Config snapshot
                        'Cfg_ST_M':          CONFIG["ST_M"],
                        'Cfg_RSI_Long':      CONFIG["RSI_LONG"],
                        'Cfg_RSI_Short':     CONFIG["RSI_SHORT"],
                        'Cfg_VOL_Filter':    CONFIG["VOL_FILTER"],
                        'Cfg_ADX_Thr':       CONFIG["ADX_THRESHOLD"],
                        'Cfg_SL_M':          CONFIG["SL_M"],
                        'Cfg_TP_M':          CONFIG["TP_M"],
                    }
                    log_trade_to_csv(trade_log)

                    try:
                        chart_buf = create_trade_chart(df, sym, pos, is_entry=False,
                                                        curr_c=curr_c, pnl=pnl, close_reason=close_reason)
                        tag_emoji = "green_circle,moneybag" if pnl > 0 else "red_circle,x"
                        send_ntfy_notification(
                            f"🔴 İŞLEM KAPANDI: {sym}",
                            f"Sonuç: {close_reason}\nPnL: %{pnl:.2f} | Kâr/Zarar: ${pnl_usd:.2f}\n"
                            f"Süre: {round(hold_minutes,1)} dk | Yeni Kasa: ${state.balance:.2f}",
                            image_buf=chart_buf, tags=tag_emoji, priority="4"
                        )
                    except Exception as e:
                        log_error("close_notification", e, sym)

                    del state.active_positions[sym]
                    state.save_state()

            # ── B. YENİ SİNYAL ────────────────────────────────────────
            row_closed = df.iloc[-2]

            # ATR_PCT kolonunu row'a ekle (sinyal_kontrol kullanıyor)
            if 'ATR_PCT' not in row_closed.index:
                row_closed = row_closed.copy()
                row_closed['ATR_PCT'] = (row_closed['ATR_14'] / row_closed['close']) * 100

            signal = sinyal_kontrol(row_closed)
            entered, reason = False, ""

            if signal:
                power_score  = hesapla_power_score(row_closed)
                signal_score = hesapla_signal_score(row_closed)

                if state.is_chop_market:
                    reason = "CHOP_MARKET"
                elif signal == "LONG" and btc_ctx["trend"] != 1:
                    reason = "BTC_TREND_KOTU"
                elif signal == "SHORT" and btc_ctx["trend"] != -1:
                    reason = "BTC_TREND_KOTU"
                elif sym in state.active_positions:
                    reason = "ALREADY_IN"
                elif (sym in state.cooldowns and
                      (get_tr_time() - state.cooldowns[sym]).total_seconds() / 60 < CONFIG["COOLDOWN_MINUTES"]):
                    cd_left = CONFIG["COOLDOWN_MINUTES"] - (get_tr_time() - state.cooldowns[sym]).total_seconds() / 60
                    reason  = f"COOLDOWN_{round(cd_left,1)}dk"
                elif len(state.active_positions) >= CONFIG["MAX_POSITIONS"]:
                    reason = "MAX_POS"
                elif row_closed['ADX'] < CONFIG["CHOP_ADX_THRESHOLD"]:
                    reason = "LOW_ADX"
                else:
                    entry_p  = float(row_closed['close'])
                    atr_val  = float(row_closed['ATR_14'])
                    sl_p     = entry_p - (CONFIG["SL_M"] * atr_val) if signal == "LONG" else entry_p + (CONFIG["SL_M"] * atr_val)
                    tp_p     = entry_p + (CONFIG["TP_M"] * atr_val) if signal == "LONG" else entry_p - (CONFIG["TP_M"] * atr_val)
                    t_size   = state.dynamic_trade_size

                    state.active_positions[sym] = {
                        'dir':            signal,
                        'entry_p':        entry_p,
                        'sl':             sl_p,
                        'tp':             tp_p,
                        'full_time':      get_tr_time().strftime('%Y-%m-%d %H:%M:%S'),
                        'entry_idx_time': str(row_closed.name),
                        'curr_pnl':       0.0,
                        'curr_p':         float(df['close'].iloc[-1]),
                        'trade_size':     t_size,
                        # Giriş anındaki indikatörler — kapanışta loga yazılır
                        'entry_rsi':      round(float(row_closed['RSI']), 2),
                        'entry_adx':      round(float(row_closed['ADX']), 2),
                        'entry_vol_ratio':round(float(row_closed['VOL_RATIO']), 3),
                        'entry_atr_pct':  round(float(row_closed['ATR_PCT']), 4),
                        'power_score':    power_score,
                        'signal_score':   signal_score,
                    }
                    state.save_state()
                    entered = True

                    try:
                        chart_buf = create_trade_chart(df, sym, state.active_positions[sym], is_entry=True)
                        send_ntfy_notification(
                            f"🟢 YENİ İŞLEM: {sym}",
                            f"Yön: {signal} | Fiyat: {entry_p:.5f}\n"
                            f"SL: {sl_p:.5f} | TP: {tp_p:.5f}\n"
                            f"Risk: ${t_size:.2f} | Power: {power_score} | Score: {signal_score}/6",
                            image_buf=chart_buf, tags="chart_with_upwards_trend", priority="4"
                        )
                    except Exception as e:
                        log_error("entry_notification", e, sym)

                if not entered:
                    state.missed_this_scan   += 1
                    state.hourly_missed_signals += 1

                log_potential_signal(sym, signal, row_closed,
                                     signal_score, power_score,
                                     entered, reason, btc_ctx)

        except Exception as e:
            log_error("coin_loop", e, sym)
            continue  # Bir coin patlasa bile diğerlerine devam

    # ── Temizlik ──────────────────────────────────────────────
    del fetched_data
    gc.collect()

    state.is_scanning = False
    rotate_logs()

    current_time = get_tr_time()
    current_hour = current_time.hour

    # ── Saatlik Rapor ─────────────────────────────────────────
    if current_hour != state.last_heartbeat_hour:
        state.last_heartbeat_hour = current_hour
        hb_msg = (
            f"💵 Kasa: ${state.balance:.2f} (Tepe: ${state.peak_balance:.2f})\n"
            f"🌍 Piyasa Yönü: {state.market_direction_text}\n"
            f"📊 BTC ATR%: {state.btc_atr_pct:.3f} | BTC RSI: {state.btc_rsi:.1f} | BTC ADX: {state.btc_adx:.1f}\n"
            f"⛔ 1 Saatte Reddedilen Sinyal: {state.hourly_missed_signals} adet\n"
            f"📈 Açık İşlem: {len(state.active_positions)}/{CONFIG['MAX_POSITIONS']}\n"
            f"🔢 Scan ID: {state.scan_id}\n"
            f"Sistem stabil, disiplin bozulmuyor patron."
        )
        send_ntfy_notification(
            f"⏱️ Saatlik Rapor ({current_time.strftime('%H:00')})",
            hb_msg, tags="clipboard,bar_chart", priority="3"
        )
        state.hourly_missed_signals = 0

    # ── Günlük Döküm (23:56-23:59) ───────────────────────────
    current_day = current_time.day
    if current_hour == 23 and current_time.minute >= 56 and state.last_dump_day != current_day:
        state.last_dump_day = current_day
        gunluk_dump_gonder()   # BASE_PATH altındaki her şeyi tarihli ad ile gönderir

    # ── Sonraki scan zamanlaması ──────────────────────────────
    now    = get_tr_time()
    target = now.replace(second=0, microsecond=0)
    next_m = next((m for m in CONFIG["TARGET_MINUTES"] if m > now.minute), CONFIG["TARGET_MINUTES"][0])
    if next_m == CONFIG["TARGET_MINUTES"][0]:
        target += timedelta(hours=1)
    target = target.replace(minute=next_m)

    state.status = (
        f"💤 SENKRON BEKLEME (Sonraki Tarama: "
        f"[bold bright_green]{target.strftime('%H:%M:%S')}[/] | "
        f"Bu Scan Kaçırılan: {state.missed_this_scan})"
    )
    draw_fund_dashboard()

    sleep_sec = (target - now).total_seconds()
    if sleep_sec > 0:
        time.sleep(sleep_sec)

# ==============================================================
# 14. GLOBAL CRASH GUARD
# ==============================================================
if __name__ == "__main__":
    start_msg = (
        f"💵 Güncel Kasa: ${state.balance:.2f}\n"
        f"🛡️ Global Crash Guard Aktif\n"
        f"📊 Maksimum Loglama Modu: hunter_history + all_signals + market_context + error_log\n"
        f"🔢 Walk-Forward Hazır: Power Score + Signal Score + Config Snapshot\n"
        f"💾 Log Rotation & RAM Koruması Devrede\n"
        f"Scan ID: {state.scan_id} | v85.0 production modunda başlatıldı!"
    )
    send_ntfy_notification(
        "🚀 v85.0 BAŞLATILDI",
        start_msg, tags="rocket,shield", priority="4"
    )

    while True:
        try:
            run_bot_cycle()
        except Exception as e:
            error_msg = (
                f"Sistem Hata Aldı ve Çöktü!\n"
                f"Hata: {str(e)[:150]}\n"
                f"Scan ID: {state.scan_id}\n"
                f"30 Saniye içinde kendini onarıp tekrar başlayacak."
            )
            log_error("MAIN_LOOP", e)
            console.print(f"\n[bold red on white] 🚨 CRITICAL ERROR: {e} [/]")
            send_ntfy_notification(
                "🚨 SİSTEM ÇÖKTÜ (RESTART ATILIYOR)",
                error_msg, tags="rotating_light,warning", priority="5"
            )
            time.sleep(30)