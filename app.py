# ====================== RBD-CRYPT v87.0 Quant Research Engine ======================
# v84.1'den v87.0'a degişiklikler:
#   - Tum 'except: pass' kaldirildi, her hata error_log.csv'ye yaziliyor
#   - MIN_ATR_PERCENT filtresi gercekten uygulaniyor
#   - score / power_score hesaplamalari gercek (walk-forward'a hazir)
#   - hunter_history.csv: 20+ yeni kolon (bar detayi, market context, indicator snapshot)
#   - all_signals.csv: genişletildi, her sinyalin tam fotografi
#   - market_context.csv: her scan'in BTC/piyasa durumu
#   - error_log.csv: sessiz hatalari yakaliyor
#   - Timeout cikişinda ST flip varsa onu bekle (MAX_HOLD + 2 bar tolerans)
#   - Cooldown sonrasi missed sinyal detayi loglaniyor
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
import threading
import signal
import atexit
import base64
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}

# ==============================================================
# 1. AYAR PANELI
# ==============================================================
CONFIG = {
    "NTFY_TOPIC": "RBD-CRYPT",
    "BASE_PATH": os.getenv("DATA_PATH", "./bot_data"),
    "MAX_POSITIONS": 3,
    "STARTING_BALANCE": 100.0,
    "FIXED_TRADE_SIZE_USD": 25.0,
    "RISK_PERCENT_PER_TRADE": 25.0,
    "MIN_TRADE_SIZE": 10.0,
    "MAX_TRADE_SIZE": 200.0,
    "TOP_COINS_LIMIT": 100,
    "CANDIDATE_TOP_COINS_LIMIT": 100,
    "ST_M": 2.8,
    "RSI_PERIOD": 9,
    # Auto-entry (seçici) eşikleri
    "AUTO_ENTRY_RSI_LONG": 62,
    "AUTO_ENTRY_RSI_SHORT": 38,
    "AUTO_ENTRY_VOL_FILTER": 1.42,
    "AUTO_ENTRY_ADX_THRESHOLD": 22,
    "AUTO_ENTRY_MIN_ATR_PERCENT": 0.85,
    "AUTO_ENTRY_MIN_POWER_SCORE": 40,
    # Candidate shortlist (gevşek) eşikleri
    "CANDIDATE_RSI_LONG": 56,
    "CANDIDATE_RSI_SHORT": 44,
    "CANDIDATE_VOL_FILTER": 1.15,
    "CANDIDATE_ADX_THRESHOLD": 17,
    "CANDIDATE_MIN_ATR_PERCENT": 0.65,
    "CANDIDATE_MIN_POWER_SCORE": 28,
    # Market policy
    "AUTO_BTC_TREND_MODE": "hard_block",   # hard_block | soft_penalty
    "AUTO_BTC_TREND_PENALTY": 12.0,
    "AUTO_CHOP_POLICY": "block",           # block | penalty | allow
    "AUTO_CHOP_PENALTY": 8.0,
    "RSI_LONG": 62,
    "RSI_SHORT": 38,
    "VOL_FILTER": 1.42,
    "ADX_THRESHOLD": 22,
    "MIN_ATR_PERCENT": 0.85,        # Artik gercekten kullaniliyor
    "SL_M": 1.65,
    "TP_M": 2.55,
    "COOLDOWN_MINUTES": 20,
    "MAX_HOLD_MINUTES": 45,
    "MAX_HOLD_ST_GRACE_BARS": 2,    # YENI: timeout dolunca ST flip icin +2 bar tolerans
    "STALE_EXIT_MIN_PNL_PCT": 0.15,     # Max hold sonrasi bu seviyenin altinda ise zayif say
    "STALE_EXIT_MIN_BEST_PNL_PCT": 0.60,# Max hold boyunca hic ivme yoksa stale exit
    "CHOP_ADX_THRESHOLD": 18,
    "BTC_VOL_THRESHOLD": 0.18,
    "MIN_POWER_SCORE": 40,
    "ENABLE_SHUTDOWN_FLUSH": True,
    "GITHUB_BACKUP_ENABLED": env_bool("GITHUB_BACKUP_ENABLED", False),
    "GITHUB_BACKUP_REPO": os.getenv("GITHUB_BACKUP_REPO", ""),          # owner/repo
    "GITHUB_BACKUP_TOKEN": os.getenv("GITHUB_BACKUP_TOKEN", ""),
    "GITHUB_BACKUP_BRANCH": os.getenv("GITHUB_BACKUP_BRANCH", "main"),
    "GITHUB_BACKUP_DIR": os.getenv("GITHUB_BACKUP_DIR", "daily-backups"),
    "TARGET_MINUTES": [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56],
    "MAX_LOG_SIZE_BYTES": 50_000_000
}

FILES = {
    "LOG":             os.path.join(CONFIG["BASE_PATH"], "hunter_history.csv"),
    "ACTIVE":          os.path.join(CONFIG["BASE_PATH"], "active_trades.json"),
    "PAPER_ACTIVE":    os.path.join(CONFIG["BASE_PATH"], "paper_active_trades.json"),
    "ALL_SIGNALS":     os.path.join(CONFIG["BASE_PATH"], "all_signals.csv"),
    "MARKET_CONTEXT":  os.path.join(CONFIG["BASE_PATH"], "market_context.csv"),   # YENI
    "ERROR_LOG":       os.path.join(CONFIG["BASE_PATH"], "error_log.csv"),         # YENI
    "STATE":           os.path.join(CONFIG["BASE_PATH"], "engine_state.json")
}

if not os.path.exists(CONFIG["BASE_PATH"]):
    os.makedirs(CONFIG["BASE_PATH"], exist_ok=True)

# ==============================================================
# 2. HATA LOGLAMA (Sessiz hatalari yakala)
# ==============================================================
def get_tr_time() -> datetime:
    return datetime.utcnow() + timedelta(hours=3)

def get_trading_day_time(now=None) -> datetime:
    """TR saatinde 03:00 cutoff ile is gunu tarihi dondurur."""
    now = now or get_tr_time()
    if now.hour < 3:
        return now - timedelta(days=1)
    return now

def get_trading_day_str(now=None) -> str:
    return get_trading_day_time(now).strftime('%Y-%m-%d')

def calc_tp_sl_metrics(entry_p: float, sl_p: float, tp_p: float, direction: str) -> tuple[float, float, float]:
    """Acilis bildirimleri icin TP/SL yuzde ve R:R hesapla."""
    entry = float(entry_p)
    sl = float(sl_p)
    tp = float(tp_p)
    if str(direction).upper() == "LONG":
        sl_pct = abs((entry - sl) / max(entry, 1e-12) * 100)
        tp_pct = abs((tp - entry) / max(entry, 1e-12) * 100)
    else:
        sl_pct = abs((sl - entry) / max(entry, 1e-12) * 100)
        tp_pct = abs((entry - tp) / max(entry, 1e-12) * 100)
    rr = tp_pct / max(sl_pct, 1e-9)
    return tp_pct, sl_pct, rr

def log_error(context: str, error: Exception, extra: str = ""):
    """Her exception'i error_log.csv'ye yazar. Hicbir hata kaybolmaz."""
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
    except Exception:
        pass  # Loglama kendisi patlarsa yapacak bir şey yok

# ==============================================================
# 3. SISTEM DURUMU
# ==============================================================
def json_safe(x):
    """numpy/pandas tiplerini JSON-serializable Python tiplerine cevirir."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x

def ascii_only(text: str) -> str:
    """HTTP header alanlari icin ASCII guvenli metin."""
    s = str(text).replace("\r", " ").replace("\n", " ")
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.strip()

class HunterState:
    def __init__(self):
        self.current_coin = "Baslatiliyor..."
        self.processed_count = 0
        self.total_count = 0
        self.status = "BASLATILIYOR"
        self.is_scanning = False
        self.cooldowns = {}
        self.market_direction_text = "Hesaplaniyor..."
        self.missed_this_scan = 0
        self.hourly_missed_signals = 0
        self.balance = CONFIG["STARTING_BALANCE"]
        self.peak_balance = CONFIG["STARTING_BALANCE"]
        self.last_heartbeat_hour = (datetime.utcnow() + timedelta(hours=3)).hour
        self.last_dump_trading_day = ""
        self.last_balance_reset_trading_day = ""
        # BTC context — her scan guncellenir, loglarda kullanilir
        self.btc_atr_pct = 0.0
        self.btc_rsi = 0.0
        self.btc_adx = 0.0
        self.btc_vol_ratio = 0.0
        self.is_chop_market = False
        self.scan_id = 0  # Her scan'e benzersiz ID, loglari birbirine baglar
        self.son_sinyaller = []   # Son 8 sinyal — dashboard'da gosterilir
        self.paper_positions = {}
        self.load_state()

    @property
    def dynamic_trade_size(self):
        fixed_size = float(CONFIG.get("FIXED_TRADE_SIZE_USD", 0) or 0)
        if fixed_size > 0:
            size = fixed_size
        else:
            size = self.balance * (CONFIG["RISK_PERCENT_PER_TRADE"] / 100.0)
        return min(max(CONFIG["MIN_TRADE_SIZE"], size), CONFIG["MAX_TRADE_SIZE"])

    def load_state(self):
        try:
            with open(FILES["ACTIVE"], 'r') as f:
                raw = json.load(f)
            # Restart sonrasi pozisyon kurtarma:
            # full_time'i şimdiki zamana sifirla ki timeout aninda tetiklenmesin.
            # Gercek giriş fiyati, SL, TP korunuyor — sadece sure sayaci resetleniyor.
            now_str = (datetime.utcnow() + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
            recovered = 0
            for sym, pos in raw.items():
                if 'entry_p' in pos and 'sl' in pos and 'tp' in pos:
                    pos['full_time']      = now_str   # sure sayacini resetle
                    pos['restarted']      = True       # log icin işaretle
                    pos.setdefault('curr_pnl', 0.0)
                    pos.setdefault('curr_p', pos['entry_p'])
                    recovered += 1
            self.active_positions = raw
            if recovered > 0:
                print(f"[RESTART] {recovered} acik pozisyon kurtarildi: {list(raw.keys())}", flush=True)
        except Exception:
            self.active_positions = {}

        try:
            with open(FILES["PAPER_ACTIVE"], 'r') as f:
                raw_paper = json.load(f)
            now_str = (datetime.utcnow() + timedelta(hours=3)).strftime('%Y-%m-%d %H:%M:%S')
            for sym, pos in raw_paper.items():
                if 'entry_p' in pos and 'sl' in pos and 'tp' in pos:
                    pos['full_time'] = now_str
                    pos['restarted'] = True
                    pos.setdefault('curr_pnl', 0.0)
                    pos.setdefault('curr_p', pos['entry_p'])
                    pos.setdefault('best_pnl', 0.0)
            self.paper_positions = raw_paper
        except Exception:
            self.paper_positions = {}

        try:
            with open(FILES["STATE"], 'r') as f:
                saved = json.load(f)
                self.balance      = saved.get("balance", CONFIG["STARTING_BALANCE"])
                self.peak_balance = saved.get("peak_balance", CONFIG["STARTING_BALANCE"])
                self.scan_id      = saved.get("scan_id", 0)
                self.last_dump_trading_day = str(saved.get("last_dump_trading_day", ""))
                self.last_balance_reset_trading_day = str(saved.get("last_balance_reset_trading_day", ""))
            print(f"[RESTART] State yuklendi — Kasa: ${self.balance:.2f} | Scan ID: {self.scan_id}", flush=True)
        except Exception:
            pass

    def save_state(self):
        try:
            state_temp = FILES["STATE"] + ".tmp"
            with open(state_temp, 'w') as f:
                json.dump({
                    "balance": float(self.balance),
                    "peak_balance": float(self.peak_balance),
                    "scan_id": int(self.scan_id),
                    "last_dump_trading_day": str(self.last_dump_trading_day or ""),
                    "last_balance_reset_trading_day": str(self.last_balance_reset_trading_day or "")
                }, f)
            os.replace(state_temp, FILES["STATE"])
            temp = FILES["ACTIVE"] + ".tmp"
            safe_positions = json_safe(self.active_positions)
            with open(temp, 'w') as f:
                json.dump(safe_positions, f)
            os.replace(temp, FILES["ACTIVE"])
            ptemp = FILES["PAPER_ACTIVE"] + ".tmp"
            safe_paper = json_safe(self.paper_positions)
            with open(ptemp, 'w') as f:
                json.dump(safe_paper, f)
            os.replace(ptemp, FILES["PAPER_ACTIVE"])
        except Exception as e:
            log_error("save_state", e)

    def update_balance(self, pnl_percent: float, trade_size: float):
        self.balance += trade_size * (pnl_percent / 100)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.save_state()

    def maybe_reset_daily_balance(self, now=None):
        """Her yeni is gununde (TR 03:00 cutoff) bakiyeyi yeniden baslat."""
        now = now or get_tr_time()
        trading_day = get_trading_day_str(now)
        if self.last_balance_reset_trading_day == trading_day:
            return False
        self.balance = float(CONFIG["STARTING_BALANCE"])
        self.peak_balance = float(CONFIG["STARTING_BALANCE"])
        self.last_balance_reset_trading_day = trading_day
        self.save_state()
        try:
            send_ntfy_notification(
                "GERCEK ISLEM | Gunluk Balance Reset",
                (
                    f"Is gunu: {trading_day} (TR cutoff 03:00)\n"
                    f"Yeni kasa: ${self.balance:.2f}\n"
                    f"Acik Gercek: {len(getattr(self, 'active_positions', {}))} | "
                    f"Acik Sanal: {len(getattr(self, 'paper_positions', {}))}"
                ),
                tags="moneybag,arrows_counterclockwise", priority="3"
            )
        except Exception as e:
            log_error("daily_balance_reset_notify", e, trading_day)
        print(f"[{now.strftime('%H:%M:%S')}] GUNLUK BALANCE RESET -> {trading_day} | ${self.balance:.2f}", flush=True)
        return True

state = HunterState()

# ==============================================================
# 4. YARDIMCI FONKSIYONLAR
# ==============================================================
def rotate_logs():
    for file_key in ["ALL_SIGNALS", "LOG", "MARKET_CONTEXT", "ERROR_LOG"]:
        path = FILES[file_key]
        try:
            if os.path.exists(path) and os.path.getsize(path) > CONFIG["MAX_LOG_SIZE_BYTES"]:
                os.rename(path, path + f"_old_{int(time.time())}")
        except Exception as e:
            log_error("rotate_logs", e, file_key)

# ==============================================================
# 5. BILDIRIM MODULU
# ==============================================================
def send_ntfy_notification(title: str, message: str, image_buf=None, tags="robot", priority="3"):
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    safe_title = ascii_only(title) or "RBD-CRYPT"
    safe_tags = ascii_only(tags) or "robot"

    headers = {
        "Title": safe_title,
        "Tags": safe_tags,
        "Priority": str(priority)
    }
    try:
        if image_buf:
            # Resim varken mesaj header'a gider; sadece ASCII kullan.
            safe_msg = ascii_only(message.replace('\n', ' | ')) or "chart"
            headers["Message"] = safe_msg
            headers["Filename"] = "chart.png"
            headers["Content-Type"] = "image/png"
            resp = requests.post(url, data=image_buf.getvalue(), headers=headers, timeout=12)
        else:
            # Resim yoksa mesaj BODY'de gider (UTF-8 direkt destekler)
            resp = requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=12)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"ntfy status={resp.status_code} body={resp.text[:200]}")
    except Exception as e:
        print(f"!!! NTFY Hatasi: {e}")
        log_error("send_ntfy_notification", e, title)

def send_ntfy_file(filepath: str, filename: str, message: str = ""):
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}"
    headers = {"Filename": (ascii_only(filename) or "file.bin")}
    if message:
        headers["Message"] = ascii_only(message.replace('\n', ' | ')) or "file"
        
    try:
        with open(filepath, 'rb') as f:
            resp = requests.put(url, data=f, headers=headers, timeout=30)
        if resp.status_code >= 400:
            raise requests.HTTPError(f"ntfy file status={resp.status_code} body={resp.text[:200]}")
    except Exception as e:
        log_error("send_ntfy_file", e, filename)

def format_hourly_report_message(current_time, real_metrics: tuple, virtual_metrics: tuple, v_pnl_usd: float) -> tuple[str, str]:
    """Telefon bildiriminde daha temiz gorunen kompakt saatlik rapor."""
    tot_trd, wins, b_wr, pf, max_dd = real_metrics
    v_tot, v_wins, v_wr, v_pf, _ = virtual_metrics
    title = f"RBD-CRYPT v87 | Saatlik {current_time.strftime('%H:00')}"
    body = (
        f"Kasa ${state.balance:.2f} | Tepe ${state.peak_balance:.2f} | Poz ${state.dynamic_trade_size:.1f}\n"
        f"Gercek {wins}/{tot_trd} | %{b_wr} | PF {pf} | DD %{max_dd:.2f}\n"
        f"Sanal {v_wins}/{v_tot} | %{v_wr} | PnL ${v_pnl_usd:.2f} | PF {v_pf}\n"
        f"BTC ATR {state.btc_atr_pct:.3f} | RSI {state.btc_rsi:.1f} | ADX {state.btc_adx:.1f}\n"
        f"Piyasa: {state.market_direction_text}\n"
        f"Acik: Gercek {len(state.active_positions)}/{CONFIG['MAX_POSITIONS']} | Sanal {len(state.paper_positions)}\n"
        f"Red(1s): {state.hourly_missed_signals} | Scan: {state.scan_id}"
    )
    return title, body

def create_daily_backup_zip(filepaths: list, tarih_str: str) -> str:
    """Gunluk dosyalari zipleyip BASE_PATH altina kaydeder."""
    zip_name = f"daily_backup_{tarih_str}.zip"
    zip_path = os.path.join(CONFIG["BASE_PATH"], zip_name)
    with zipfile.ZipFile(zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in filepaths:
            if not os.path.exists(fp):
                continue
            # Ayni zip'i tekrar ziplemeye calisirsak OSError olabilir.
            if os.path.abspath(fp) == os.path.abspath(zip_path):
                continue
            arcname = os.path.relpath(fp, CONFIG["BASE_PATH"]).replace("\\", "/")
            try:
                zf.write(fp, arcname=arcname)
            except OSError as e:
                log_error("create_daily_backup_zip_item", e, fp)
                continue
    return zip_path

def upload_backup_to_github(zip_path: str, tarih_str: str) -> tuple[bool, str]:
    """
    GitHub private repo backup (Contents API).
    Donus: (success, info_msg)
    """
    if not CONFIG.get("GITHUB_BACKUP_ENABLED", False):
        return False, "disabled"
    repo = str(CONFIG.get("GITHUB_BACKUP_REPO", "")).strip()
    token = str(CONFIG.get("GITHUB_BACKUP_TOKEN", "")).strip()
    branch = str(CONFIG.get("GITHUB_BACKUP_BRANCH", "main")).strip() or "main"
    folder = str(CONFIG.get("GITHUB_BACKUP_DIR", "daily-backups")).strip().strip("/")
    if not repo or not token:
        return False, "missing_repo_or_token"
    try:
        with open(zip_path, "rb") as f:
            content_b64 = base64.b64encode(f.read()).decode("ascii")
        remote_name = os.path.basename(zip_path)
        remote_path = f"{folder}/{remote_name}" if folder else remote_name
        url = f"https://api.github.com/repos/{repo}/contents/{remote_path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "RBD-CRYPT-BACKUP"
        }
        payload = {
            "message": f"backup: {tarih_str}",
            "content": content_b64,
            "branch": branch
        }
        r = requests.put(url, headers=headers, json=payload, timeout=60)
        if r.status_code in (200, 201):
            return True, f"{repo}/{remote_path}@{branch}"
        return False, f"http_{r.status_code}"
    except Exception as e:
        log_error("upload_backup_to_github", e, os.path.basename(zip_path))
        return False, type(e).__name__


def gunluk_dump_gonder():
    """
    BASE_PATH altindaki TuM dosyalari (ve bir seviye alt klasorleri) ntfy'ye gonderir.
    Her dosyanin adina tarih eki eklenir:  hunter_history_2026-02-23.csv
    JSON ve CSV'ler sirayla, aralarinda 1s beklenerek gonderilir (spam engeli).
    """
    tarih_str = get_trading_day_str()
    base      = CONFIG["BASE_PATH"]

    # BASE_PATH altindaki tum dosyalari topla (recursive, _old_ arşivleri dahil degil)
    dosyalar = []
    for root, dirs, files in os.walk(base):
        # _old_ arşivleri ve temp dosyalari atla
        files = [
            f for f in files
            if not f.endswith('.tmp')
            and '_old_' not in f
            and not f.startswith('daily_backup_')
        ]
        for fname in sorted(files):
            dosyalar.append(os.path.join(root, fname))

    if not dosyalar:
        send_ntfy_notification(
            f"📦 Gunluk Dokum ({tarih_str})",
            "BASE_PATH icinde gonderilecek dosya bulunamadi.",
            tags="warning", priority="3"
        )
        return

    # Spam azaltma: tek mesaj (tek ZIP dosya bildirimi)
    toplam_kb = round(sum(os.path.getsize(d) for d in dosyalar if os.path.exists(d)) / 1024, 1)
    zip_path = ""
    backup_note = "GitHub backup: kapali"
    try:
        zip_path = create_daily_backup_zip(dosyalar, tarih_str)
        zip_size_kb = round(os.path.getsize(zip_path) / 1024, 1) if os.path.exists(zip_path) else 0
        ok, info = upload_backup_to_github(zip_path, tarih_str)
        if ok:
            backup_note = f"GitHub backup OK -> {info} ({zip_size_kb} KB)"
        else:
            backup_note = f"GitHub backup yok ({info}) | local zip: {os.path.basename(zip_path)} ({zip_size_kb} KB)"
    except Exception as e:
        log_error("daily_zip_backup", e, tarih_str)
        backup_note = f"GitHub backup hata: {type(e).__name__} | {str(e)[:120]}"

    try:
        zip_size_kb = round(os.path.getsize(zip_path) / 1024, 1) if (zip_path and os.path.exists(zip_path)) else 0
        if zip_path and os.path.exists(zip_path):
            send_ntfy_file(
                zip_path,
                os.path.basename(zip_path),
                f"GUNLUK DOKUM | tarih={tarih_str} | dosya={len(dosyalar)} | toplam={toplam_kb} KB | zip={zip_size_kb} KB | {backup_note}"
            )
        else:
            send_ntfy_notification(
                f"GUNLUK DOKUM ({tarih_str})",
                f"ZIP olusmadi. Dosya sayisi: {len(dosyalar)}\n{backup_note}",
                tags="warning,package", priority="3"
            )
    except Exception as e:
        log_error("gunluk_dump_gonder_zip_ntfy", e, tarih_str)

# ==============================================================
# 6. VERI cEKIMI
# ==============================================================
def safe_api_get(url: str, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=10)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                time.sleep(10)  # Rate limit — bekle
            else:
                time.sleep(2)
        except requests.exceptions.RequestException as e:
            log_error("safe_api_get", e, f"url={url} attempt={attempt}")
            time.sleep(3)
    return None

def get_top_futures_coins(limit=30) -> list:
    data = safe_api_get("https://fapi.binance.com/fapi/v1/ticker/24hr")
    if data:
        def is_ascii_clean(s):
            try: s.encode('ascii'); return True
            except UnicodeEncodeError: return False

        usdt_pairs = [
            d for d in data
            if d['symbol'].endswith('USDT')
            and '_' not in d['symbol']
            and is_ascii_clean(d['symbol'])   # cince/unicode karakterli coin filtresi
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
# 7. INDIKAToRLER
# ==============================================================
def _talib_ready_array(series: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(series, errors='coerce').to_numpy(dtype=np.float64, copy=False)
    return np.ascontiguousarray(arr, dtype=np.float64)

def hesapla_indikatorler(df: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    # talib float64 zorunlu — Binance bazen object/float32 doner, cast şart
    c = _talib_ready_array(df['close'])
    h = _talib_ready_array(df['high'])
    l = _talib_ready_array(df['low'])
    v = _talib_ready_array(df['volume'])
    if min(len(c), len(h), len(l), len(v)) < 50:
        raise ValueError("insufficient bars after numeric conversion")
    if (not np.isfinite(c[-50:]).all() or not np.isfinite(h[-50:]).all() or
        not np.isfinite(l[-50:]).all() or not np.isfinite(v[-50:]).all()):
        raise ValueError("non-finite values in recent OHLCV")

    df['RSI']       = talib.RSI(c, CONFIG["RSI_PERIOD"])
    df['ADX']       = talib.ADX(h, l, c, 14)
    df['PLUS_DI']   = talib.PLUS_DI(h, l, c, 14)   # YENI: DI farki data icin
    df['MINUS_DI']  = talib.MINUS_DI(h, l, c, 14)  # YENI
    df['EMA20']     = talib.EMA(c, 20)
    df['EMA50']     = talib.EMA(c, 50)              # YENI: trend derinligi
    df['ATR_14']    = talib.ATR(h, l, c, 14)
    df['BBANDS_UP'], df['BBANDS_MID'], df['BBANDS_LOW'] = talib.BBANDS(c, 20, 2, 2)  # YENI
    df['MACD'], df['MACD_SIGNAL'], df['MACD_HIST'] = talib.MACD(c, 12, 26, 9)        # YENI
    df['VOL_SMA_20'] = talib.SMA(v, 20)
    df['VOL_RATIO']  = np.where(df['VOL_SMA_20'] > 0, v / df['VOL_SMA_20'], 0)
    df['ATR_PCT']    = (df['ATR_14'] / df['close']) * 100                             # YENI: normalize ATR

    # Supertrend
    atr_st = talib.ATR(h, l, c, 10)
    hl2    = (h + l) / 2
    st_line = np.zeros(len(c))
    trend   = np.ones(len(c))

    for i in range(1, len(c)):
        if np.isnan(atr_st[i]) or np.isnan(hl2[i]):
            # ATR henuz hesaplanamadi (ilk N bar) — onceki degeri koru
            st_line[i] = st_line[i-1]
            trend[i]   = trend[i-1]
            continue
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
    df['ST_DIST_PCT'] = ((df['close'] - df['ST_LINE']) / df['close']) * 100  # YENI
    # FLIP: sadece bir onceki barda yon degişimi — daha temiz sinyal
    df['FLIP_LONG']  = (df['TREND'] == 1) & (df['TREND'].shift(1) == -1)
    df['FLIP_SHORT'] = (df['TREND'] == -1) & (df['TREND'].shift(1) == 1)

    # Price action
    df['BODY_PCT']   = abs(df['close'] - df['open']) / df['open'] * 100      # YENI
    df['UPPER_WICK'] = (df['high'] - df[['close','open']].max(axis=1)) / df['open'] * 100  # YENI
    df['LOWER_WICK'] = (df[['close','open']].min(axis=1) - df['low']) / df['open'] * 100   # YENI

    return df

# ==============================================================
# 8. SINYAL VE SKOR HESABI
# ==============================================================
def hesapla_power_score(row, thresholds: dict = None) -> float:
    """
    0-100 arasi sinyal gucu skoru.
    Walk-forward analizinde hangi skorlarin karli oldugunu gormek icin kullanilir.
    Alt bileşenler de loglaniyor.
    """
    score = 0.0
    thresholds = thresholds or get_signal_thresholds("auto")

    # RSI bileşeni (0-25)
    if bool(row['FLIP_LONG']):
        rsi_component = max(0, min(25, (float(row['RSI']) - thresholds["rsi_long"]) * 2.5))
    else:
        rsi_component = max(0, min(25, (thresholds["rsi_short"] - float(row['RSI'])) * 2.5))
    score += rsi_component

    # Hacim bileşeni (0-25)
    vol_component = max(0, min(25, (float(row['VOL_RATIO']) - thresholds["vol_filter"]) * 15))
    score += vol_component

    # ADX bileşeni (0-20)
    adx_component = max(0, min(20, (float(row['ADX']) - thresholds["adx_threshold"]) * 0.8))
    score += adx_component

    # ATR % bileşeni (0-15) — duşuk ATR duşuk skor
    atr_component = max(0, min(15, (float(row['ATR_PCT']) - thresholds["min_atr_pct"]) * 5))
    score += atr_component

    # MACD histogram yonu (0-10)
    if bool(row['FLIP_LONG']) and float(row['MACD_HIST']) > 0:
        score += 10
    elif bool(row['FLIP_SHORT']) and float(row['MACD_HIST']) < 0:
        score += 10

    # BB genişligi (0-5) — sikişik piyasayi cezalandir
    bb_width = (float(row['BBANDS_UP']) - float(row['BBANDS_LOW'])) / max(float(row['BBANDS_MID']), 1e-10) * 100
    bb_component = max(0, min(5, bb_width * 0.5))
    score += bb_component

    return round(score, 2)

def hesapla_signal_score(row, signal_type: str = None, thresholds: dict = None) -> int:
    """
    Kac koşul saglandi (0-6). Basit sayim, gucu degil adeti verir.
    """
    thresholds = thresholds or get_signal_thresholds("auto")
    if signal_type is None:
        signal_type = get_flip_candidate_signal(row)
    return score_from_flags(evaluate_signal_filters(row, signal_type, thresholds))

def get_flip_candidate_signal(row):
    """Aday sinyal: yalnizca FLIP'e gore LONG/SHORT doner."""
    flip_long = bool(row['FLIP_LONG'])
    flip_short = bool(row['FLIP_SHORT'])
    if flip_long and not flip_short:
        return "LONG"
    if flip_short and not flip_long:
        return "SHORT"
    if flip_long and flip_short:
        # Nadir edge-case: son trend yonune gore sec
        return "LONG" if int(row.get('TREND', 0)) >= 0 else "SHORT"
    return None

def get_candidate_fail_reason(row, candidate_signal: str) -> str:
    """FLIP var ama teknik filtreler tam degilse ilk patlayan kosulu yazar."""
    t = get_signal_thresholds("candidate")
    is_long = candidate_signal == "LONG"
    atr_pct = (float(row['ATR_14']) / max(float(row['close']), 1e-10)) * 100
    if is_long and not (float(row['RSI']) > t["rsi_long"]):
        return "CAND_FAIL_RSI"
    if (not is_long) and not (float(row['RSI']) < t["rsi_short"]):
        return "CAND_FAIL_RSI"
    if not (float(row['VOL_RATIO']) > t["vol_filter"]):
        return "CAND_FAIL_VOL"
    if is_long and not (float(row['close']) > float(row['EMA20'])):
        return "CAND_FAIL_EMA"
    if (not is_long) and not (float(row['close']) < float(row['EMA20'])):
        return "CAND_FAIL_EMA"
    if not (float(row['ADX']) > t["adx_threshold"]):
        return "CAND_FAIL_ADX"
    if not (atr_pct >= t["min_atr_pct"]):
        return "CAND_FAIL_ATR"
    return "CAND_FAIL_OTHER"

def get_signal_thresholds(layer: str = "auto") -> dict:
    if layer == "candidate":
        return {
            "rsi_long": float(CONFIG["CANDIDATE_RSI_LONG"]),
            "rsi_short": float(CONFIG["CANDIDATE_RSI_SHORT"]),
            "vol_filter": float(CONFIG["CANDIDATE_VOL_FILTER"]),
            "adx_threshold": float(CONFIG["CANDIDATE_ADX_THRESHOLD"]),
            "min_atr_pct": float(CONFIG["CANDIDATE_MIN_ATR_PERCENT"]),
            "min_power_score": float(CONFIG["CANDIDATE_MIN_POWER_SCORE"]),
        }
    return {
        "rsi_long": float(CONFIG["AUTO_ENTRY_RSI_LONG"]),
        "rsi_short": float(CONFIG["AUTO_ENTRY_RSI_SHORT"]),
        "vol_filter": float(CONFIG["AUTO_ENTRY_VOL_FILTER"]),
        "adx_threshold": float(CONFIG["AUTO_ENTRY_ADX_THRESHOLD"]),
        "min_atr_pct": float(CONFIG["AUTO_ENTRY_MIN_ATR_PERCENT"]),
        "min_power_score": float(CONFIG["AUTO_ENTRY_MIN_POWER_SCORE"]),
    }

def evaluate_signal_filters(row, signal_type: str, thresholds: dict) -> dict:
    if signal_type not in ("LONG", "SHORT"):
        return {
            "flip_ok": False, "rsi_ok": False, "vol_ok": False, "adx_ok": False,
            "atr_ok": False, "ema_ok": False, "all_ok": False
        }
    is_long = signal_type == "LONG"
    flip_ok = bool(row['FLIP_LONG']) if is_long else bool(row['FLIP_SHORT'])
    rsi_ok = float(row['RSI']) > thresholds["rsi_long"] if is_long else float(row['RSI']) < thresholds["rsi_short"]
    vol_ok = float(row['VOL_RATIO']) > thresholds["vol_filter"]
    adx_ok = float(row['ADX']) > thresholds["adx_threshold"]
    atr_ok = (float(row['ATR_14']) / max(float(row['close']), 1e-10) * 100) >= thresholds["min_atr_pct"]
    ema_ok = float(row['close']) > float(row['EMA20']) if is_long else float(row['close']) < float(row['EMA20'])
    all_ok = all([flip_ok, rsi_ok, vol_ok, adx_ok, atr_ok, ema_ok])
    return {
        "flip_ok": flip_ok, "rsi_ok": bool(rsi_ok), "vol_ok": bool(vol_ok),
        "adx_ok": bool(adx_ok), "atr_ok": bool(atr_ok), "ema_ok": bool(ema_ok),
        "all_ok": bool(all_ok)
    }

def score_from_flags(flags: dict) -> int:
    return int(sum(1 for k in ["flip_ok", "rsi_ok", "vol_ok", "adx_ok", "atr_ok", "ema_ok"] if flags.get(k)))

def sinyal_kontrol(row):
    auto_t = get_signal_thresholds("auto")
    for sig in ("LONG", "SHORT"):
        if evaluate_signal_filters(row, sig, auto_t)["all_ok"]:
            return sig
    return None

# ==============================================================
# 9. BTC ANALIZI
# ==============================================================
def get_btc_context() -> dict:
    """BTC'den tam bir context sozlugu doner — hem filtre hem loglama icin."""
    btc_df = get_live_futures_data("BTCUSDT", 200)
    if btc_df is None or len(btc_df) < 50:
        return {
            "trend": 0, "atr_pct": 0.0, "rsi": 0.0, "adx": 0.0,
            "vol_ratio": 0.0, "macd_hist": 0.0, "close": 0.0,
            "ema20": 0.0, "bb_width_pct": 0.0
        }
    try:
        btc_df = hesapla_indikatorler(btc_df, "BTCUSDT")
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
        try:
            dtypes_str = ",".join(f"{c}:{btc_df[c].dtype}" for c in ['open','high','low','close','volume'] if c in btc_df.columns)
        except Exception:
            dtypes_str = "dtype_unavailable"
        log_error("get_btc_context", e, dtypes_str)
        return {
            "trend": 0, "atr_pct": 0.0, "rsi": 0.0, "adx": 0.0,
            "vol_ratio": 0.0, "macd_hist": 0.0, "close": 0.0,
            "ema20": 0.0, "bb_width_pct": 0.0
        }

# ==============================================================
# 10. LOGLAMA
# ==============================================================
def log_trade_to_csv(trade_dict: dict):
    """Kapanan işlemi hunter_history.csv'ye yazar. Append mode — tum dosyayi okumaz."""
    try:
        pd.DataFrame([trade_dict]).to_csv(
            FILES["LOG"], mode='a',
            header=not os.path.exists(FILES["LOG"]),
            index=False, encoding='utf-8-sig'
        )
    except Exception as e:
        log_error("log_trade_to_csv", e)

def log_potential_signal(sym: str, signal_type: str, row, score: int,
                          power_score: float, entered: bool, reason: str = "",
                          btc_ctx: dict = None, candidate_signal: str = None,
                          candidate_flags: dict = None, auto_flags: dict = None,
                          decision_meta: dict = None):
    """
    all_signals.csv — Her tespit edilen sinyalin tam fotografi.
    Girişe donuşsun ya da donuşmesin, tum koşullar kaydediliyor.
    Walk-forward: 'tradable=False' ama karli olan sinyaller
    parametre ayari icin altin madeni.
    """
    if btc_ctx is None:
        btc_ctx = {}
    candidate_flags = candidate_flags or {}
    auto_flags = auto_flags or {}
    decision_meta = decision_meta or {}

    try:
        bb_width = (float(row['BBANDS_UP']) - float(row['BBANDS_LOW'])) / max(float(row['BBANDS_MID']), 1e-10) * 100
        log_row = {
            # Zaman / kimlik
            'timestamp':        get_tr_time().isoformat(),
            'scan_id':          state.scan_id,
            'coin':             sym,
            'signal':           signal_type,
            'is_candidate':     bool(candidate_signal),
            'candidate_signal': candidate_signal or "",
            # Sinyal kalitesi
            'score':            score,
            'power_score':      power_score,
            # Fiyat
            'close':            round(float(row['close']), 6),
            'open':             round(float(row['open']), 6),
            'high':             round(float(row['high']), 6),
            'low':              round(float(row['low']), 6),
            'volume':           round(float(row['volume']), 2),
            # Indikatorler
            'rsi':              round(float(row['RSI']), 2),
            'adx':              round(float(row['ADX']), 2),
            'plus_di':          round(float(row['PLUS_DI']), 2),
            'minus_di':         round(float(row['MINUS_DI']), 2),
            'atr_14':           round(float(row['ATR_14']), 6),
            'atr_pct':          round(float(row['ATR_PCT']), 4),
            'vol_ratio':        round(float(row['VOL_RATIO']), 3),
            'ema20':            round(float(row['EMA20']), 6),
            'ema50':            round(float(row['EMA50']), 6),
            'st_line':          round(float(row['ST_LINE']), 6),
            'st_dist_pct':      round(float(row['ST_DIST_PCT']), 4),
            'macd':             round(float(row['MACD']), 6),
            'macd_signal':      round(float(row['MACD_SIGNAL']), 6),
            'macd_hist':        round(float(row['MACD_HIST']), 6),
            'bb_upper':         round(float(row['BBANDS_UP']), 6),
            'bb_lower':         round(float(row['BBANDS_LOW']), 6),
            'bb_width_pct':     round(float(bb_width), 3),
            # Price action
            'body_pct':         round(float(row['BODY_PCT']), 4),
            'upper_wick_pct':   round(float(row['UPPER_WICK']), 4),
            'lower_wick_pct':   round(float(row['LOWER_WICK']), 4),
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
            # Aday ve auto-entry filtre flaglari
            'flip_long':        bool(row.get('FLIP_LONG', False)),
            'flip_short':       bool(row.get('FLIP_SHORT', False)),
            'rsi_ok':           bool(candidate_flags.get('rsi_ok', False)),
            'vol_ok':           bool(candidate_flags.get('vol_ok', False)),
            'adx_ok':           bool(candidate_flags.get('adx_ok', False)),
            'atr_ok':           bool(candidate_flags.get('atr_ok', False)),
            'ema_ok':           bool(candidate_flags.get('ema_ok', False)),
            'power_ok':         bool(decision_meta.get('power_ok', False)),
            'btc_trend_ok':     bool(decision_meta.get('btc_trend_ok', False)),
            'chop_ok':          bool(decision_meta.get('chop_ok', True)),
            'cooldown_ok':      bool(decision_meta.get('cooldown_ok', True)),
            'capacity_ok':      bool(decision_meta.get('capacity_ok', True)),
            'already_in':       bool(decision_meta.get('already_in', False)),
            'auto_entry_eligible': bool(decision_meta.get('auto_entry_eligible', False)),
            'rejection_stage':  str(decision_meta.get('rejection_stage', "")),
            'rejection_reason': str(decision_meta.get('rejection_reason', reason)),
            'auto_rsi_ok':      bool(auto_flags.get('rsi_ok', False)),
            'auto_vol_ok':      bool(auto_flags.get('vol_ok', False)),
            'auto_adx_ok':      bool(auto_flags.get('adx_ok', False)),
            'auto_atr_ok':      bool(auto_flags.get('atr_ok', False)),
            'auto_ema_ok':      bool(auto_flags.get('ema_ok', False)),
            # Karar
            'tradable':         entered,
            'blocked_reason':   reason,
            # Filtre eşikleri (parametre degişirse retroaktif analiz icin)
            'cfg_rsi_long':     CONFIG["AUTO_ENTRY_RSI_LONG"],
            'cfg_rsi_short':    CONFIG["AUTO_ENTRY_RSI_SHORT"],
            'cfg_vol_filter':   CONFIG["AUTO_ENTRY_VOL_FILTER"],
            'cfg_adx_thr':      CONFIG["AUTO_ENTRY_ADX_THRESHOLD"],
            'cfg_atr_min_pct':  CONFIG["AUTO_ENTRY_MIN_ATR_PERCENT"],
            'cfg_cand_rsi_long':CONFIG["CANDIDATE_RSI_LONG"],
            'cfg_cand_rsi_short':CONFIG["CANDIDATE_RSI_SHORT"],
            'cfg_cand_vol_filter':CONFIG["CANDIDATE_VOL_FILTER"],
            'cfg_cand_adx_thr': CONFIG["CANDIDATE_ADX_THRESHOLD"],
            'cfg_cand_atr_min_pct': CONFIG["CANDIDATE_MIN_ATR_PERCENT"],
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
    market_context.csv — Her scan'in başindaki piyasa snapshot'i.
    Hangi market koşullarinda ne kadar sinyal uretildigini analiz etmek icin.
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

def get_advanced_metrics(trade_mode: str = "REAL"):
    try:
        if not os.path.exists(FILES["LOG"]):
            return 0, 0, 0, 0.0, 0.0
        df = pd.read_csv(FILES["LOG"])
        if len(df) == 0:
            return 0, 0, 0, 0.0, 0.0
        if 'Trade_Mode' in df.columns:
            if trade_mode:
                tm = df['Trade_Mode'].astype(str).str.upper()
                if str(trade_mode).upper() == "REAL":
                    df = df[(tm == "REAL") | tm.isin(["", "NAN", "NONE"])].copy()
                else:
                    df = df[tm == str(trade_mode).upper()].copy()
        else:
            # Eski loglar varsayılan olarak GERÇEK kabul edilir
            if str(trade_mode).upper() == "VIRTUAL":
                return 0, 0, 0, 0.0, 0.0
        if len(df) == 0:
            return 0, 0, 0, 0.0, 0.0
        wins   = df[df['PnL_Yuzde'] > 0]
        losses = df[df['PnL_Yuzde'] <= 0]
        tot_trd, w_count = len(df), len(wins)
        gross_profit = wins['PnL_USD'].sum()   if 'PnL_USD' in wins.columns   else wins['PnL_Yuzde'].sum()
        gross_loss   = abs(losses['PnL_USD'].sum()) if 'PnL_USD' in losses.columns else abs(losses['PnL_Yuzde'].sum())
        pf  = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 99.9
        mdd = calc_real_mdd_from_trade_log(df) if str(trade_mode).upper() == "REAL" else 0.0
        return tot_trd, w_count, int((w_count / tot_trd) * 100) if tot_trd > 0 else 0, pf, mdd
    except Exception as e:
        log_error("get_advanced_metrics", e)
        return 0, 0, 0, 0.0, 0.0

def calc_real_mdd_from_trade_log(df: pd.DataFrame) -> float:
    """Gercek MDD: kapanan trade equity curve'u uzerinden hesaplanir."""
    try:
        if df is None or len(df) == 0:
            return 0.0

        if 'Kasa_Son_Durum' in df.columns:
            equity = pd.to_numeric(df['Kasa_Son_Durum'], errors='coerce').dropna()
        elif 'PnL_USD' in df.columns:
            pnl_usd = pd.to_numeric(df['PnL_USD'], errors='coerce').fillna(0.0)
            equity = pnl_usd.cumsum() + float(CONFIG["STARTING_BALANCE"])
        else:
            return round(((state.peak_balance - state.balance) / max(state.peak_balance, 1e-9)) * 100, 2)

        if len(equity) == 0:
            return 0.0

        running_peak = equity.cummax()
        dd_pct = ((running_peak - equity) / running_peak.replace(0, np.nan)) * 100
        return round(float(dd_pct.fillna(0.0).max()), 2)
    except Exception as e:
        log_error("calc_real_mdd_from_trade_log", e)
        return round(((state.peak_balance - state.balance) / max(state.peak_balance, 1e-9)) * 100, 2)

# ==============================================================
# 11. GRAFIK
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
        # Bar genişligini fiyat araligina gore dinamik ayarla
        bar_w = (plot_df.index[-1] - plot_df.index[0]).total_seconds() / len(plot_df) * 0.6 / 86400
        ax.bar(up.index,   up.close   - up.open,   bar_w, bottom=up.open,     color='#2ecc71', alpha=0.9)
        ax.bar(down.index, down.open  - down.close, bar_w, bottom=down.close, color='#e74c3c', alpha=0.9)

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

def maybe_open_paper_position(sym: str, df: pd.DataFrame, row_closed, signal: str,
                              power_score: float, signal_score: int, reason: str = ""):
    """Aday shortlist için sanal işlem açar (aynı sembolde bir tane aktif)."""
    try:
        if not signal:
            return
        if sym in state.paper_positions:
            return
        entry_p = float(row_closed['close'])
        atr_val = float(row_closed['ATR_14'])
        sl_p = entry_p - (CONFIG["SL_M"] * atr_val) if signal == "LONG" else entry_p + (CONFIG["SL_M"] * atr_val)
        tp_p = entry_p + (CONFIG["TP_M"] * atr_val) if signal == "LONG" else entry_p - (CONFIG["TP_M"] * atr_val)
        v_size = state.dynamic_trade_size

        state.paper_positions[sym] = {
            'dir': signal,
            'entry_p': entry_p,
            'sl': sl_p,
            'tp': tp_p,
            'full_time': get_tr_time().strftime('%Y-%m-%d %H:%M:%S'),
            'entry_idx_time': str(row_closed.name),
            'curr_pnl': 0.0,
            'best_pnl': 0.0,
            'curr_p': float(df['close'].iloc[-1]),
            'trade_size': v_size,
            'entry_rsi': round(float(row_closed['RSI']), 2),
            'entry_adx': round(float(row_closed['ADX']), 2),
            'entry_vol_ratio': round(float(row_closed['VOL_RATIO']), 3),
            'entry_atr_pct': round(float(row_closed['ATR_PCT']), 4),
            'power_score': float(power_score),
            'signal_score': int(signal_score),
            'candidate_reason': reason or "SHORTLIST",
        }
        state.save_state()

        try:
            chart_buf = create_trade_chart(df, sym, state.paper_positions[sym], is_entry=True)
            tp_pct, sl_pct, rr = calc_tp_sl_metrics(entry_p, sl_p, tp_p, signal)
            send_ntfy_notification(
                f"SANAL ISLEM ACILDI: {sym}",
                f"Yon: {signal} | Fiyat: {entry_p:.5f}\n"
                f"SL: {sl_p:.5f} (-%{sl_pct:.2f}) | TP: {tp_p:.5f} (+%{tp_pct:.2f})\n"
                f"Hedef/Risk (R:R): {rr:.2f}\n"
                f"Pozisyon(sim): ${v_size:.2f} | Power: {power_score:.1f} | Score: {signal_score}/6\n"
                f"Neden: {reason or 'SHORTLIST'}",
                image_buf=chart_buf, tags="mag,chart_with_upwards_trend", priority="3"
            )
        except Exception as e:
            log_error("paper_entry_notification", e, sym)
    except Exception as e:
        log_error("maybe_open_paper_position", e, sym)

def update_paper_position_for_symbol(sym: str, df: pd.DataFrame, btc_ctx: dict):
    """Sanal işlemi TP/SL/stale kurallarıyla yönetir, kapanırsa log+ntfy gönderir."""
    if sym not in state.paper_positions:
        return
    try:
        pos = state.paper_positions[sym]
        curr_h = float(df['high'].iloc[-1])
        curr_l = float(df['low'].iloc[-1])
        curr_c = float(df['close'].iloc[-1])
        entry_p = float(pos.get('entry_p', curr_c))

        pos['curr_p'] = curr_c
        closed = False
        close_reason = ""
        pnl = 0.0

        pos_time = datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')
        hold_minutes = (get_tr_time() - pos_time).total_seconds() / 60

        curr_pnl_live = ((curr_c - entry_p) / entry_p * 100 if pos['dir'] == 'LONG'
                         else (entry_p - curr_c) / entry_p * 100)
        pos['curr_pnl'] = curr_pnl_live
        pos['best_pnl'] = max(float(pos.get('best_pnl', curr_pnl_live)), curr_pnl_live)

        if pos['dir'] == 'LONG':
            if curr_h >= pos['tp']:
                pnl = (pos['tp'] - entry_p) / entry_p * 100
                closed, close_reason = True, "KAR ALDI"
            elif curr_l <= pos['sl']:
                pnl = (pos['sl'] - entry_p) / entry_p * 100
                closed, close_reason = True, "STOP OLDU"
        else:
            if curr_l <= pos['tp']:
                pnl = (entry_p - pos['tp']) / entry_p * 100
                closed, close_reason = True, "KAR ALDI"
            elif curr_h >= pos['sl']:
                pnl = (entry_p - pos['sl']) / entry_p * 100
                closed, close_reason = True, "STOP OLDU"

        if (not closed) and hold_minutes > CONFIG["MAX_HOLD_MINUTES"]:
            last_trend = int(df['TREND'].iloc[-1])
            grace_limit = CONFIG["MAX_HOLD_MINUTES"] + CONFIG["MAX_HOLD_ST_GRACE_BARS"] * 5
            adverse_flip = ((pos['dir'] == 'LONG' and last_trend == -1) or
                            (pos['dir'] == 'SHORT' and last_trend == 1))
            if curr_pnl_live > 0:
                if pos['dir'] == 'LONG':
                    pos['sl'] = max(float(pos['sl']), float(entry_p))
                else:
                    pos['sl'] = min(float(pos['sl']), float(entry_p))
            if adverse_flip:
                closed, close_reason, pnl = True, "TREND_FLIP_EXIT", curr_pnl_live
            elif hold_minutes > grace_limit:
                best_pnl = float(pos.get('best_pnl', curr_pnl_live))
                ema20_now = float(df['EMA20'].iloc[-1])
                ema_against = ((pos['dir'] == 'LONG' and curr_c < ema20_now) or
                               (pos['dir'] == 'SHORT' and curr_c > ema20_now))
                no_progress = (curr_pnl_live <= float(CONFIG["STALE_EXIT_MIN_PNL_PCT"]) and
                               best_pnl < float(CONFIG["STALE_EXIT_MIN_BEST_PNL_PCT"]))
                if no_progress or ema_against:
                    closed, close_reason, pnl = True, "STALE_EXIT", curr_pnl_live

        if not closed:
            return

        trade_size = float(pos.get('trade_size', state.dynamic_trade_size))
        pnl_usd = trade_size * (pnl / 100)
        row_at_close = df.iloc[-2]
        trade_log = {
            'Trade_Mode':         'VIRTUAL',
            'Scan_ID':            state.scan_id,
            'Restarted':          pos.get('restarted', False),
            'Tarih':              get_trading_day_str(),
            'Giris_Saati':        pos['full_time'].split(' ')[1],
            'Cikis_Saati':        get_tr_time().strftime('%H:%M:%S'),
            'Hold_Dakika':        round(hold_minutes, 1),
            'Coin':               sym,
            'Yon':                pos['dir'],
            'Giris_Fiyati':       round(entry_p, 6),
            'Cikis_Fiyati':       round(curr_c, 6),
            'TP_Seviyesi':        round(pos['tp'], 6),
            'SL_Seviyesi':        round(pos['sl'], 6),
            'TP_SL_Orani':        round(abs(pos['tp'] - entry_p) / max(abs(pos['sl'] - entry_p), 1e-12), 3),
            'Risk_USD':           round(trade_size, 2),
            'PnL_Yuzde':          round(pnl, 2),
            'PnL_USD':            round(pnl_usd, 2),
            'Kasa_Son_Durum':     round(state.balance, 2),  # sanal işlem bakiyeyi değiştirmez
            'Sonuc':              close_reason,
            'Giris_RSI':          pos.get('entry_rsi', 0),
            'Giris_ADX':          pos.get('entry_adx', 0),
            'Giris_VOL_RATIO':    pos.get('entry_vol_ratio', 0),
            'Giris_ATR_PCT':      pos.get('entry_atr_pct', 0),
            'Giris_Power_Score':  pos.get('power_score', 0),
            'Giris_Score':        pos.get('signal_score', 0),
            'Cikis_RSI':          round(float(row_at_close['RSI']), 2),
            'Cikis_ADX':          round(float(row_at_close['ADX']), 2),
            'Cikis_VOL_RATIO':    round(float(row_at_close['VOL_RATIO']), 3),
            'Cikis_ATR_PCT':      round(float(row_at_close['ATR_PCT']), 4),
            'Cikis_TREND':        int(row_at_close['TREND']),
            'Cikis_MACD_HIST':    round(float(row_at_close.get('MACD_HIST', 0)), 6),
            'BTC_Trend':          btc_ctx.get('trend', 0),
            'BTC_ATR_PCT':        btc_ctx.get('atr_pct', 0.0),
            'BTC_RSI':            btc_ctx.get('rsi', 0.0),
            'BTC_ADX':            btc_ctx.get('adx', 0.0),
            'BTC_Vol_Ratio':      btc_ctx.get('vol_ratio', 0.0),
        }
        log_trade_to_csv(trade_log)

        try:
            chart_buf = create_trade_chart(df, sym, pos, is_entry=False, curr_c=curr_c, pnl=pnl, close_reason=close_reason)
            send_ntfy_notification(
                f"SANAL ISLEM KAPANDI: {sym}",
                f"Sonuc: {close_reason}\nPnL: %{pnl:.2f} | Sim PnL: ${pnl_usd:.2f}\n"
                f"Sure: {round(hold_minutes,1)} dk | Yon: {pos['dir']}\n"
                f"Giris: {entry_p:.5f} | Cikis: {curr_c:.5f}",
                image_buf=chart_buf, tags="clipboard,chart_with_upwards_trend", priority="3"
            )
        except Exception as e:
            log_error("paper_close_notification", e, sym)

        del state.paper_positions[sym]
        state.save_state()
    except Exception as e:
        log_error("update_paper_position_for_symbol", e, sym)

# ==============================================================
# 12. DASHBOARD
# ==============================================================
def log_print(msg: str):
    """Zaman damgali duz print — Railway loglarinda okunmasi kolay."""
    zaman = get_tr_time().strftime('%H:%M:%S')
    print(f"[{zaman}] {msg}", flush=True)

_shutdown_flag = {"done": False}

def log_storage_diagnostics():
    """Startup'ta persistence path ve state dosyalarini raporla."""
    try:
        log_print(f"STORAGE PATH: {CONFIG['BASE_PATH']}")
        for key in ["ACTIVE", "STATE"]:
            p = FILES[key]
            exists = os.path.exists(p)
            size = os.path.getsize(p) if exists else 0
            log_print(f"  {key}: {'OK' if exists else 'MISSING'} | {p} | {size} bytes")
        log_print(f"  Recovered active positions: {len(getattr(state, 'active_positions', {}))}")
        log_print(f"  Recovered paper positions: {len(getattr(state, 'paper_positions', {}))}")
        log_print(
            f"  GitHub backup: {'ENABLED' if CONFIG.get('GITHUB_BACKUP_ENABLED') else 'DISABLED'} | "
            f"Repo: {CONFIG.get('GITHUB_BACKUP_REPO','-')} | Branch: {CONFIG.get('GITHUB_BACKUP_BRANCH','main')}"
        )
    except Exception as e:
        log_error("log_storage_diagnostics", e)

def flush_state_on_shutdown(reason: str = "shutdown"):
    if _shutdown_flag["done"]:
        return
    _shutdown_flag["done"] = True
    try:
        if CONFIG.get("ENABLE_SHUTDOWN_FLUSH", True):
            state.save_state()
            log_print(f"STATE FLUSH OK ({reason})")
    except Exception as e:
        log_error("flush_state_on_shutdown", e, reason)

def _handle_termination(signum, frame):
    try:
        log_print(f"TERM SIGNAL ALINDI: {signum}")
        flush_state_on_shutdown(f"signal_{signum}")
        try:
            send_ntfy_notification(
                "BOT SHUTDOWN",
                f"Signal: {signum}\nScan ID: {state.scan_id}\nAcik pozisyon: {len(state.active_positions)}",
                tags="warning", priority="3"
            )
        except Exception as e:
            log_error("shutdown_ntfy", e, str(signum))
    finally:
        raise SystemExit(0)


def draw_fund_dashboard():
    """Her 10 coinde bir cagrilir. Duz log satirlari yazar."""
    tot_trd, wins, b_wr, pf, max_dd = get_advanced_metrics()
    kasa_ok = "+" if state.balance >= CONFIG["STARTING_BALANCE"] else "-"

    print("-" * 70, flush=True)
    log_print(f"RBD-CRYPT v87.0 | Scan #{state.scan_id}")
    log_print(f"PIYASA : {state.market_direction_text} | BTC ATR%: {state.btc_atr_pct:.3f} | BTC RSI: {state.btc_rsi:.1f} | BTC ADX: {state.btc_adx:.1f}")
    log_print(f"KASA   : ${state.balance:.2f} ({kasa_ok}) | Tepe: ${state.peak_balance:.2f} | Pozisyon/islem: ${state.dynamic_trade_size:.1f}")
    log_print(f"PERFORMANS: {wins}/{tot_trd} islem (%{b_wr} basari) | PF: {pf} | Max DD: %{max_dd:.2f}")

    # Aktif pozisyonlar
    if state.active_positions:
        log_print(f"ACIK POZISYONLAR ({len(state.active_positions)}/{CONFIG['MAX_POSITIONS']}):")
        for sym, pos in state.active_positions.items():
            curr_pnl = pos.get('curr_pnl', 0.0)
            dur = str(get_tr_time() - datetime.strptime(pos['full_time'], '%Y-%m-%d %H:%M:%S')).split('.')[0]
            yon  = "LONG" if pos.get('dir') == 'LONG' else "SHORT"
            isaret = "+" if curr_pnl >= 0 else ""
            log_print(f"  >> {sym} {yon} | Sure: {dur} | PnL: {isaret}{curr_pnl:.2f}% | Giris: {pos.get('entry_p',0):.5f} | TP: {pos.get('tp',0):.5f} | SL: {pos.get('sl',0):.5f}")
    else:
        log_print(f"ACIK POZISYON: Yok (0/{CONFIG['MAX_POSITIONS']})")

    # Son sinyaller
    if state.son_sinyaller:
        log_print(f"SON SINYALLER (son {min(8, len(state.son_sinyaller))}):")
        for s in state.son_sinyaller[-8:]:
            durum = "GIRILDI" if s["entered"] else f"REDDEDILDI({s['reason']})"
            log_print(f"  {s['zaman']} {s['coin']:15s} {s['signal']:5s} | {durum:30s} | Power:{s['power']:5.0f} RSI:{s['rsi']:5.1f} ADX:{s['adx']:5.1f} VOL:{s['vol']:.2f}x")
    else:
        log_print("SON SINYALLER: Henuz sinyal tespit edilmedi.")

    # Tarama ilerlemesi
    if state.is_scanning:
        total  = state.total_count if state.total_count > 0 else 1
        done   = state.processed_count
        pct    = int((done / total) * 100)
        filled = int(20 * done / total)
        bar    = "#" * filled + "." * (20 - filled)
        log_print(f"TARAMA : [{bar}] %{pct} ({done}/{total}) | Simdi: {state.current_coin}")
    else:
        log_print(f"DURUM  : {state.status}")
    print("-" * 70, flush=True)

# ==============================================================
# NTFY KOMUT DINLEYICISI
# ==============================================================
def ntfy_komut_dinle():
    """
    Ayri thread'de calişir. ntfy subscribe endpoint'ini dinler.
    Desteklenen komutlar (ntfy'den mesaj olarak gonder):
      logs   — tum log dosyalarini hemen gonderir
      durum  — anlik kasa/pozisyon ozeti
      status — durum ile ayni (alias)
    """
    url = f"https://ntfy.sh/{CONFIG['NTFY_TOPIC']}/sse"

    while True:
        try:
            with requests.get(url, stream=True, timeout=(10, 90)) as resp:  # (connect, read) timeout
                for line in resp.iter_lines():
                    if not line:
                        continue
                    line = line.decode('utf-8', errors='ignore')
                    if not line.startswith('data:'):
                        continue
                    try:
                        payload = json.loads(line[5:].strip())
                        # Botun kendi gonderdigi bildirimleri yoksay
                        # (kendi mesajina tepki vermesin)
                        baslik = payload.get('title', '')
                        if any(x in baslik for x in ['ISLEM', 'BASLATILDI', 'coKTu', 
                                                      'Rapor', 'Dokum', 'RESTART', 'Durum']):
                            continue
                        mesaj = payload.get('message', '').strip().lower()
                        if not mesaj:
                            continue

                        log_print(f"NTFY KOMUT ALINDI: {mesaj}")

                        if mesaj == 'logs':
                            send_ntfy_notification(
                                "📦 Manuel Log Talebi Alindi",
                                "Dosyalar hazirlaniyor, birazdan gelecek...",
                                tags="package", priority="3"
                            )
                            threading.Thread(target=gunluk_dump_gonder, daemon=True).start()

                        elif mesaj in ('durum', 'status'):
                            tot_trd, wins, b_wr, pf, max_dd = get_advanced_metrics()
                            acik = len(state.active_positions)
                            pozlar = ""
                            for sym, pos in state.active_positions.items():
                                pozlar += f"  • {sym} {pos['dir']} | PnL: %{pos.get('curr_pnl',0):.2f}\n"
                            durum_msg = (
                                f"💵 Kasa: ${state.balance:.2f} (Tepe: ${state.peak_balance:.2f})\n"
                                f"📈 Acik Işlem: {acik}/{CONFIG['MAX_POSITIONS']}\n"
                                f"{pozlar}"
                                f"🏆 Başari: {wins}/{tot_trd} (%{b_wr}) | PF: {pf}\n"
                                f"🌍 Piyasa: {state.market_direction_text}\n"
                                f"🔢 Scan ID: {state.scan_id}"
                            )
                            send_ntfy_notification(
                                f"📊 Anlik Durum ({get_tr_time().strftime('%H:%M')})",
                                durum_msg, tags="bar_chart", priority="3"
                            )

                    except Exception as e:
                        log_error("ntfy_komut_parse", e, line[:100])

        except Exception as e:
            log_error("ntfy_komut_dinle", e)
            time.sleep(15)   # baglanti koparsa 15sn bekle yeniden baglan



# ==============================================================
# 13. ANA KONTROL DoNGuSu
# ==============================================================
def run_bot_cycle():
    state.scan_id += 1
    state.save_state()

    state.status = "🌐 BINANCE FUTURES: Hacimli Coinler cekiliyor..."
    draw_fund_dashboard()
    scan_limit = int(max(CONFIG["TOP_COINS_LIMIT"], CONFIG.get("CANDIDATE_TOP_COINS_LIMIT", CONFIG["TOP_COINS_LIMIT"])))
    coins = get_top_futures_coins(scan_limit)

    # --- BTC Context ---
    btc_ctx = get_btc_context()
    state.btc_atr_pct   = btc_ctx["atr_pct"]
    state.btc_rsi       = btc_ctx["rsi"]
    state.btc_adx       = btc_ctx["adx"]
    state.btc_vol_ratio = btc_ctx["vol_ratio"]

    state.is_chop_market = btc_ctx["atr_pct"] < CONFIG["BTC_VOL_THRESHOLD"]

    if state.is_chop_market:
        state.market_direction_text = "CHOP MARKET (Dusuk Volatilite)"
    else:
        state.market_direction_text = "YUKSELIS (LONG)" if btc_ctx["trend"] == 1 else "DUSUS (SHORT)"

    # Piyasa context'ini logla
    log_market_context(btc_ctx, len(coins), len(state.active_positions))

    state.total_count = len(coins)
    state.processed_count = 0
    state.is_scanning = True
    state.missed_this_scan = 0
    state.status = "🚀 QUANT MOTORU: VADELI PIYASA TARANIYOR (Asenkron)..."

    # --- Asenkron veri cekimi ---
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

    # --- Coin dongusu ---
    for sym in coins:
        try:
            df = fetched_data.get(sym)
            state.current_coin  = sym
            state.processed_count += 1
            # Her 10 coinde bir ciz — Railway 500 log/sn limitini aşmamak icin
            if state.processed_count % 10 == 1 or state.processed_count == state.total_count:
                draw_fund_dashboard()

            if df is None or len(df) < 50:
                continue

            try:
                df = hesapla_indikatorler(df, sym)
            except Exception as e:
                try:
                    dtypes_str = ",".join(f"{c}:{df[c].dtype}" for c in ['open','high','low','close','volume'] if c in df.columns)
                    extra = f"{sym} | {dtypes_str}"
                except Exception:
                    extra = sym
                log_error("hesapla_indikatorler", e, extra)
                continue

            # ── A. AKTIF ISLEM KONTROLu ──────────────────────────────
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

                curr_pnl_live = ((curr_c - entry_p) / entry_p * 100 if pos['dir'] == 'LONG'
                                 else (entry_p - curr_c) / entry_p * 100)
                pos['curr_pnl'] = curr_pnl_live
                pos['best_pnl'] = max(float(pos.get('best_pnl', curr_pnl_live)), curr_pnl_live)

                # TP/SL her zaman once kontrol edilir
                if pos['dir'] == 'LONG':
                    if curr_h >= pos['tp']:
                        pnl = (pos['tp'] - entry_p) / entry_p * 100
                        closed, close_reason = True, "KAR ALDI"
                    elif curr_l <= pos['sl']:
                        pnl = (pos['sl'] - entry_p) / entry_p * 100
                        closed, close_reason = True, "STOP OLDU"
                else:
                    if curr_l <= pos['tp']:
                        pnl = (entry_p - pos['tp']) / entry_p * 100
                        closed, close_reason = True, "KAR ALDI"
                    elif curr_h >= pos['sl']:
                        pnl = (entry_p - pos['sl']) / entry_p * 100
                        closed, close_reason = True, "STOP OLDU"

                # Timeout yerine akilli stale/momentum exit
                if (not closed) and hold_minutes > CONFIG["MAX_HOLD_MINUTES"]:
                    last_trend = int(df['TREND'].iloc[-1])
                    grace_limit = CONFIG["MAX_HOLD_MINUTES"] + CONFIG["MAX_HOLD_ST_GRACE_BARS"] * 5
                    adverse_flip = ((pos['dir'] == 'LONG' and last_trend == -1) or
                                    (pos['dir'] == 'SHORT' and last_trend == 1))

                    if curr_pnl_live > 0:
                        if pos['dir'] == 'LONG':
                            pos['sl'] = max(float(pos['sl']), float(entry_p))
                        else:
                            pos['sl'] = min(float(pos['sl']), float(entry_p))

                    if adverse_flip:
                        closed = True
                        close_reason = "TREND_FLIP_EXIT"
                        pnl = curr_pnl_live
                    elif hold_minutes > grace_limit:
                        best_pnl = float(pos.get('best_pnl', curr_pnl_live))
                        ema20_now = float(df['EMA20'].iloc[-1])
                        ema_against = ((pos['dir'] == 'LONG' and curr_c < ema20_now) or
                                       (pos['dir'] == 'SHORT' and curr_c > ema20_now))
                        no_progress = (curr_pnl_live <= float(CONFIG["STALE_EXIT_MIN_PNL_PCT"]) and
                                       best_pnl < float(CONFIG["STALE_EXIT_MIN_BEST_PNL_PCT"]))
                        if no_progress or ema_against:
                            closed = True
                            close_reason = "STALE_EXIT"
                            pnl = curr_pnl_live

                if closed:
                    state.cooldowns[sym] = get_tr_time()
                    trade_size = pos.get('trade_size', CONFIG["MIN_TRADE_SIZE"])
                    pnl_usd    = trade_size * (pnl / 100)
                    state.update_balance(pnl, trade_size)

                    # ── Geniş kapaniş logu ────────────────────────────
                    row_at_close = df.iloc[-2]
                    trade_log = {
                        # Kimlik
                        'Trade_Mode':         'REAL',
                        'Scan_ID':           state.scan_id,
                        'Restarted':         pos.get('restarted', False),
                        'Tarih':             get_trading_day_str(),
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
                        # Giriş anindaki indikatorler (pos icinden)
                        'Giris_RSI':         pos.get('entry_rsi', 0),
                        'Giris_ADX':         pos.get('entry_adx', 0),
                        'Giris_VOL_RATIO':   pos.get('entry_vol_ratio', 0),
                        'Giris_ATR_PCT':     pos.get('entry_atr_pct', 0),
                        'Giris_Power_Score': pos.get('power_score', 0),
                        'Giris_Score':       pos.get('signal_score', 0),
                        # cikiş anindaki indikatorler
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
                            f"GERCEK ISLEM KAPANDI: {sym}",
                            f"Sonuc: {close_reason}\nPnL: %{pnl:.2f} | Kâr/Zarar: ${pnl_usd:.2f}\n"
                            f"Sure: {round(hold_minutes,1)} dk | Yeni Kasa: ${state.balance:.2f}",
                            image_buf=chart_buf, tags=tag_emoji, priority="4"
                        )
                    except Exception as e:
                        log_error("close_notification", e, sym)

                    del state.active_positions[sym]
                    state.save_state()
                    continue  # Ayni scan'de re-entry engeli

            if sym in state.paper_positions:
                update_paper_position_for_symbol(sym, df, btc_ctx)

            # ── B. YENI SINYAL ────────────────────────────────────────
            row_closed = df.iloc[-2]

            candidate_signal = get_flip_candidate_signal(row_closed)
            signal = None
            entered, reason = False, ""

            if candidate_signal:
                cand_t = get_signal_thresholds("candidate")
                auto_t = get_signal_thresholds("auto")
                candidate_flags = evaluate_signal_filters(row_closed, candidate_signal, cand_t)
                auto_flags = evaluate_signal_filters(row_closed, candidate_signal, auto_t)
                power_score = float(hesapla_power_score(row_closed, cand_t))
                auto_power_score = float(hesapla_power_score(row_closed, auto_t))
                signal_score = int(hesapla_signal_score(row_closed, candidate_signal, cand_t))
                auto_signal_score = int(hesapla_signal_score(row_closed, candidate_signal, auto_t))
                log_signal_type = candidate_signal
                shortlist_eligible = False
                decision_meta = {
                    'power_ok': power_score >= cand_t["min_power_score"],
                    'btc_trend_ok': True,
                    'chop_ok': True,
                    'cooldown_ok': True,
                    'capacity_ok': True,
                    'already_in': False,
                    'auto_entry_eligible': False,
                    'rejection_stage': '',
                    'rejection_reason': ''
                }

                if not candidate_flags["all_ok"]:
                    reason = get_candidate_fail_reason(row_closed, candidate_signal)
                    decision_meta['rejection_stage'] = "candidate_filter"
                    decision_meta['rejection_reason'] = reason
                elif power_score < cand_t["min_power_score"]:
                    reason = f"CAND_LOW_POWER_{power_score:.0f}"
                    decision_meta['power_ok'] = False
                    decision_meta['rejection_stage'] = "candidate_filter"
                    decision_meta['rejection_reason'] = reason
                else:
                    shortlist_eligible = True
                    signal = candidate_signal if auto_flags["all_ok"] else None
                    log_signal_type = signal if signal else candidate_signal
                    if signal is None:
                        reason = "AUTO_TECH_FAIL"
                        decision_meta['rejection_stage'] = "auto_filter"
                        decision_meta['rejection_reason'] = reason
                    else:
                        effective_auto_power = auto_power_score
                        btc_trend_match = False
                        if btc_ctx["trend"] != 0:
                            btc_trend_match = ((signal == "LONG" and btc_ctx["trend"] == 1) or
                                               (signal == "SHORT" and btc_ctx["trend"] == -1))
                        decision_meta['btc_trend_ok'] = bool(btc_trend_match)

                        if CONFIG.get("AUTO_BTC_TREND_MODE", "hard_block") == "soft_penalty":
                            if not btc_trend_match:
                                effective_auto_power -= float(CONFIG.get("AUTO_BTC_TREND_PENALTY", 0.0))
                        else:
                            if btc_ctx["trend"] == 0:
                                reason = "BTC_VERI_YOK"
                            elif not btc_trend_match:
                                reason = "BTC_TREND_KOTU"
                            if reason:
                                decision_meta['rejection_stage'] = "market_filter"
                                decision_meta['rejection_reason'] = reason

                        if not reason and state.is_chop_market:
                            chop_policy = str(CONFIG.get("AUTO_CHOP_POLICY", "block")).lower()
                            if chop_policy == "block":
                                decision_meta['chop_ok'] = False
                                reason = "CHOP_MARKET"
                                decision_meta['rejection_stage'] = "market_filter"
                                decision_meta['rejection_reason'] = reason
                            elif chop_policy == "penalty":
                                effective_auto_power -= float(CONFIG.get("AUTO_CHOP_PENALTY", 0.0))

                        if not reason and effective_auto_power < auto_t["min_power_score"]:
                            reason = f"LOW_POWER_{effective_auto_power:.0f}"
                            decision_meta['power_ok'] = False
                            decision_meta['rejection_stage'] = "auto_filter"
                            decision_meta['rejection_reason'] = reason

                        if not reason and sym in state.active_positions:
                            reason = "ALREADY_IN"
                            decision_meta['already_in'] = True
                            decision_meta['rejection_stage'] = "execution_filter"
                            decision_meta['rejection_reason'] = reason

                        if (not reason and sym in state.cooldowns and
                              (get_tr_time() - state.cooldowns[sym]).total_seconds() / 60 < CONFIG["COOLDOWN_MINUTES"]):
                            cd_left = CONFIG["COOLDOWN_MINUTES"] - (get_tr_time() - state.cooldowns[sym]).total_seconds() / 60
                            reason  = f"COOLDOWN_{round(cd_left,1)}dk"
                            decision_meta['cooldown_ok'] = False
                            decision_meta['rejection_stage'] = "execution_filter"
                            decision_meta['rejection_reason'] = reason

                        if not reason and len(state.active_positions) >= CONFIG["MAX_POSITIONS"]:
                            reason = "MAX_POS"
                            decision_meta['capacity_ok'] = False
                            decision_meta['rejection_stage'] = "execution_filter"
                            decision_meta['rejection_reason'] = reason

                        if not reason and float(row_closed['ADX']) < float(CONFIG["CHOP_ADX_THRESHOLD"]):
                            reason = "LOW_ADX"
                            decision_meta['rejection_stage'] = "execution_filter"
                            decision_meta['rejection_reason'] = reason

                        if not reason:
                            decision_meta['auto_entry_eligible'] = True
                            power_score = effective_auto_power
                            signal_score = auto_signal_score

                if decision_meta.get('auto_entry_eligible', False):
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
                        'best_pnl':       0.0,
                        'curr_p':         float(df['close'].iloc[-1]),
                        'trade_size':     t_size,
                        # Giriş anindaki indikatorler — kapanişta loga yazilir
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
                        tp_pct, sl_pct, rr = calc_tp_sl_metrics(entry_p, sl_p, tp_p, signal)
                        send_ntfy_notification(
                            f"GERCEK ISLEM ACILDI: {sym}",
                            f"Yon: {signal} | Fiyat: {entry_p:.5f}\n"
                            f"SL: {sl_p:.5f} (-%{sl_pct:.2f}) | TP: {tp_p:.5f} (+%{tp_pct:.2f})\n"
                            f"Hedef/Risk (R:R): {rr:.2f}\n"
                            f"Pozisyon: ${t_size:.2f} | Power: {power_score} | Score: {signal_score}/6",
                            image_buf=chart_buf, tags="chart_with_upwards_trend", priority="4"
                        )
                    except Exception as e:
                        log_error("entry_notification", e, sym)

                if shortlist_eligible:
                    maybe_open_paper_position(
                        sym=sym, df=df, row_closed=row_closed, signal=candidate_signal,
                        power_score=float(hesapla_power_score(row_closed, cand_t)),
                        signal_score=int(hesapla_signal_score(row_closed, candidate_signal, cand_t)),
                        reason=reason if reason else "SHORTLIST_OK"
                    )

                if signal and not entered:
                    state.missed_this_scan      += 1
                    state.hourly_missed_signals += 1

                if signal:
                    # Dashboard icin son sinyalleri tut (max 50)
                    state.son_sinyaller.append({
                        'zaman':   get_tr_time().strftime('%H:%M:%S'),
                        'coin':    sym,
                        'signal':  signal,
                        'entered': entered,
                        'reason':  reason if not entered else '-',
                        'power':   power_score,
                        'rsi':     round(float(row_closed['RSI']), 1),
                        'adx':     round(float(row_closed['ADX']), 1),
                        'vol':     round(float(row_closed['VOL_RATIO']), 2),
                    })
                    if len(state.son_sinyaller) > 50:
                        state.son_sinyaller = state.son_sinyaller[-50:]

                # Adaylar (flip) dahil tum sinyalleri logla
                log_potential_signal(sym, log_signal_type, row_closed,
                                     signal_score, power_score,
                                     entered, reason, btc_ctx,
                                     candidate_signal=candidate_signal,
                                     candidate_flags=candidate_flags,
                                     auto_flags=auto_flags,
                                     decision_meta=decision_meta)

        except Exception as e:
            log_error("coin_loop", e, sym)
            continue  # Bir coin patlasa bile digerlerine devam

    # ── Temizlik ──────────────────────────────────────────────
    del fetched_data
    gc.collect()

    state.is_scanning = False
    rotate_logs()

    current_time = get_tr_time()
    current_hour = current_time.hour
    state.maybe_reset_daily_balance(current_time)

    # ── Saatlik Rapor ─────────────────────────────────────────
    if current_hour != state.last_heartbeat_hour:
        state.last_heartbeat_hour = current_hour
        
        # Dashboard'daki performans metriklerini cekiyoruz
        tot_trd, wins, b_wr, pf, max_dd = get_advanced_metrics("REAL")
        v_tot, v_wins, v_wr, v_pf, _ = get_advanced_metrics("VIRTUAL")
        v_pnl_usd = 0.0
        try:
            if os.path.exists(FILES["LOG"]):
                _h = pd.read_csv(FILES["LOG"])
                if 'Trade_Mode' in _h.columns:
                    _h = _h[_h['Trade_Mode'].astype(str).str.upper() == 'VIRTUAL']
                else:
                    _h = _h.iloc[0:0]
                if 'PnL_USD' in _h.columns and len(_h) > 0:
                    v_pnl_usd = float(pd.to_numeric(_h['PnL_USD'], errors='coerce').fillna(0.0).sum())
        except Exception as e:
            log_error("hourly_virtual_metrics", e)
        
        hb_title, hb_msg = format_hourly_report_message(
            current_time=current_time,
            real_metrics=(tot_trd, wins, b_wr, pf, max_dd),
            virtual_metrics=(v_tot, v_wins, v_wr, v_pf, 0.0),
            v_pnl_usd=v_pnl_usd
        )
        send_ntfy_notification(
            hb_title,
            hb_msg, tags="clipboard,bar_chart", priority="3"
        )
        state.hourly_missed_signals = 0

    # ── Gunluk Dokum (23:56-23:59) ───────────────────────────
    trading_day = get_trading_day_str(current_time)
    if current_hour == 2 and current_time.minute >= 56 and state.last_dump_trading_day != trading_day:
        state.last_dump_trading_day = trading_day
        state.save_state()
        gunluk_dump_gonder()   # BASE_PATH altindaki her şeyi tarihli ad ile gonderir

    # ── Sonraki scan zamanlamasi ──────────────────────────────
    now    = get_tr_time()
    target = now.replace(second=0, microsecond=0)
    next_m = next((m for m in CONFIG["TARGET_MINUTES"] if m > now.minute), CONFIG["TARGET_MINUTES"][0])
    if next_m == CONFIG["TARGET_MINUTES"][0]:
        target += timedelta(hours=1)
    target = target.replace(minute=next_m)

    state.status = (
        f"💤 SENKRON BEKLEME (Sonraki Tarama: "
        f"{target.strftime('%H:%M:%S')} | "
        f"Bu Scan Kacirilan: {state.missed_this_scan})"
    )
    draw_fund_dashboard()

    sleep_sec = (target - now).total_seconds()
    if sleep_sec > 0:
        time.sleep(sleep_sec)

# ==============================================================
# 14. GLOBAL CRASH GUARD
# ==============================================================
if __name__ == "__main__":
    atexit.register(lambda: flush_state_on_shutdown("atexit"))
    try:
        signal.signal(signal.SIGTERM, _handle_termination)
        signal.signal(signal.SIGINT, _handle_termination)
    except Exception as e:
        log_error("signal_handler_setup", e)
    log_storage_diagnostics()

    start_msg = (
        f"💵 Guncel Kasa: ${state.balance:.2f}\n"
        f"🛡️ Global Crash Guard Aktif\n"
        f"📊 Maksimum Loglama Modu: hunter_history + all_signals + market_context + error_log\n"
        f"🔢 Walk-Forward Hazir: Power Score + Signal Score + Config Snapshot\n"
        f"💾 Log Rotation & RAM Korumasi Devrede\n"
        f"Scan ID: {state.scan_id} | v87.0 production modunda başlatildi!"
    )
    log_print("=" * 50)
    log_print("RBD-CRYPT v87.0 BASLATILDI")
    log_print(f"Kasa: ${state.balance:.2f} | Scan ID: {state.scan_id}")
    log_print("=" * 50)
    send_ntfy_notification(
        "🚀 v87.0 BASLATILDI",
        start_msg, tags="rocket,shield", priority="4"
    )

    # NTFY komut dinleyiciyi ayri thread'de başlat
    komut_thread = threading.Thread(target=ntfy_komut_dinle, daemon=True)
    komut_thread.start()

    # Restart sonrasi acik pozisyonlari bildir
    if state.active_positions:
        pozlar = ", ".join(
            f"{sym} {pos['dir']} @ {pos['entry_p']:.5f}"
            for sym, pos in state.active_positions.items()
        )
        send_ntfy_notification(
            "🔄 RESTART — Pozisyonlar Kurtarildi",
            f"{len(state.active_positions)} acik pozisyon devam ediyor:\n{pozlar}\nSure sayaci resetlendi.",
            tags="arrows_counterclockwise,white_check_mark", priority="4"
        )
    if state.paper_positions:
        spoz = ", ".join(
            f"{sym} {pos['dir']} @ {pos['entry_p']:.5f}"
            for sym, pos in state.paper_positions.items()
        )
        send_ntfy_notification(
            "RESTART - SANAL POZISYONLAR",
            f"{len(state.paper_positions)} acik sanal pozisyon devam ediyor:\n{spoz}\nSure sayaci resetlendi.",
            tags="arrows_counterclockwise,mag", priority="3"
        )

    while True:
        try:
            run_bot_cycle()
        except Exception as e:
            error_msg = (
                f"Sistem Hata Aldi ve coktu!\n"
                f"Hata: {str(e)[:150]}\n"
                f"Scan ID: {state.scan_id}\n"
                f"30 Saniye icinde kendini onarip tekrar başlayacak."
            )
            log_error("MAIN_LOOP", e)
            log_print(f"CRITICAL ERROR: {e}")
            send_ntfy_notification(
                "🚨 SISTEM coKTu (RESTART ATILIYOR)",
                error_msg, tags="rotating_light,warning", priority="5"
            )
            time.sleep(30)
