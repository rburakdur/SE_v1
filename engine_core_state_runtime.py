from datetime import datetime, timedelta

import json


def env_bool(name: str, default: bool = False) -> bool:
    import os

    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def get_tr_time() -> datetime:
    return datetime.utcnow() + timedelta(hours=3)


def get_trading_day_time(now=None) -> datetime:
    now = now or get_tr_time()
    if now.hour < 3:
        return now - timedelta(days=1)
    return now


def get_trading_day_str(now=None) -> str:
    return get_trading_day_time(now).strftime("%Y-%m-%d")


def calc_tp_sl_metrics(entry_p: float, sl_p: float, tp_p: float, direction: str) -> tuple[float, float, float]:
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
    import app as A

    try:
        row = {
            "timestamp": get_tr_time().isoformat(),
            "context": context,
            "error_type": type(error).__name__,
            "error_msg": str(error)[:300],
            "traceback": A.traceback.format_exc()[-500:],
            "extra": extra,
        }
        A.pd.DataFrame([row]).to_csv(
            A.FILES["ERROR_LOG"],
            mode="a",
            header=not A.os.path.exists(A.FILES["ERROR_LOG"]),
            index=False,
        )
    except Exception:
        pass


def json_safe(x):
    import app as A

    if isinstance(x, (A.np.integer,)):
        return int(x)
    if isinstance(x, (A.np.floating,)):
        return float(x)
    if isinstance(x, (A.np.bool_,)):
        return bool(x)
    if isinstance(x, (A.pd.Timestamp, datetime)):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def ascii_only(text: str) -> str:
    s = str(text).replace("\r", " ").replace("\n", " ")
    s = s.encode("ascii", "ignore").decode("ascii")
    return s.strip()


class HunterState:
    def __init__(self):
        import app as A

        self.current_coin = "Baslatiliyor..."
        self.processed_count = 0
        self.total_count = 0
        self.status = "BASLATILIYOR"
        self.is_scanning = False
        self.cooldowns = {}
        self.market_direction_text = "Hesaplaniyor..."
        self.missed_this_scan = 0
        self.hourly_missed_signals = 0
        self.balance = A.CONFIG["STARTING_BALANCE"]
        self.peak_balance = A.CONFIG["STARTING_BALANCE"]
        self.last_heartbeat_hour = (datetime.utcnow() + timedelta(hours=3)).hour
        self.last_dump_trading_day = ""
        self.last_balance_reset_trading_day = ""
        self.btc_atr_pct = 0.0
        self.btc_rsi = 0.0
        self.btc_adx = 0.0
        self.btc_vol_ratio = 0.0
        self.is_chop_market = False
        self.scan_id = 0
        self.son_sinyaller = []
        self.paper_positions = {}
        self.load_state()

    @property
    def dynamic_trade_size(self):
        import app as A

        fixed_size = float(A.CONFIG.get("FIXED_TRADE_SIZE_USD", 0) or 0)
        if fixed_size > 0:
            size = fixed_size
        else:
            size = self.balance * (A.CONFIG["RISK_PERCENT_PER_TRADE"] / 100.0)
        return min(max(A.CONFIG["MIN_TRADE_SIZE"], size), A.CONFIG["MAX_TRADE_SIZE"])

    def load_state(self):
        import app as A

        try:
            with open(A.FILES["ACTIVE"], "r") as f:
                raw = json.load(f)
            now_str = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
            recovered = 0
            for sym, pos in raw.items():
                if "entry_p" in pos and "sl" in pos and "tp" in pos:
                    pos["full_time"] = now_str
                    pos["restarted"] = True
                    pos.setdefault("curr_pnl", 0.0)
                    pos.setdefault("curr_p", pos["entry_p"])
                    recovered += 1
            self.active_positions = raw
            if recovered > 0:
                print(f"[RESTART] {recovered} acik pozisyon kurtarildi: {list(raw.keys())}", flush=True)
        except Exception:
            self.active_positions = {}

        try:
            with open(A.FILES["PAPER_ACTIVE"], "r") as f:
                raw_paper = json.load(f)
            now_str = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")
            for sym, pos in raw_paper.items():
                if "entry_p" in pos and "sl" in pos and "tp" in pos:
                    pos["full_time"] = now_str
                    pos["restarted"] = True
                    pos.setdefault("curr_pnl", 0.0)
                    pos.setdefault("curr_p", pos["entry_p"])
                    pos.setdefault("best_pnl", 0.0)
            self.paper_positions = raw_paper
        except Exception:
            self.paper_positions = {}

        try:
            with open(A.FILES["STATE"], "r") as f:
                saved = json.load(f)
                self.balance = saved.get("balance", A.CONFIG["STARTING_BALANCE"])
                self.peak_balance = saved.get("peak_balance", A.CONFIG["STARTING_BALANCE"])
                self.scan_id = saved.get("scan_id", 0)
                self.last_dump_trading_day = str(saved.get("last_dump_trading_day", ""))
                self.last_balance_reset_trading_day = str(saved.get("last_balance_reset_trading_day", ""))
            print(f"[RESTART] State yuklendi â€” Kasa: ${self.balance:.2f} | Scan ID: {self.scan_id}", flush=True)
        except Exception:
            pass

    def save_state(self):
        import app as A

        try:
            state_temp = A.FILES["STATE"] + ".tmp"
            with open(state_temp, "w") as f:
                json.dump(
                    {
                        "balance": float(self.balance),
                        "peak_balance": float(self.peak_balance),
                        "scan_id": int(self.scan_id),
                        "last_dump_trading_day": str(self.last_dump_trading_day or ""),
                        "last_balance_reset_trading_day": str(self.last_balance_reset_trading_day or ""),
                    },
                    f,
                )
            A.os.replace(state_temp, A.FILES["STATE"])

            temp = A.FILES["ACTIVE"] + ".tmp"
            with open(temp, "w") as f:
                json.dump(A.json_safe(self.active_positions), f)
            A.os.replace(temp, A.FILES["ACTIVE"])

            ptemp = A.FILES["PAPER_ACTIVE"] + ".tmp"
            with open(ptemp, "w") as f:
                json.dump(A.json_safe(self.paper_positions), f)
            A.os.replace(ptemp, A.FILES["PAPER_ACTIVE"])
        except Exception as e:
            A.log_error("save_state", e)

    def update_balance(self, pnl_percent: float, trade_size: float):
        self.balance += trade_size * (pnl_percent / 100)
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        self.save_state()

    def maybe_reset_daily_balance(self, now=None):
        import app as A

        now = now or get_tr_time()
        trading_day = get_trading_day_str(now)
        if self.last_balance_reset_trading_day == trading_day:
            return False
        self.balance = float(A.CONFIG["STARTING_BALANCE"])
        self.peak_balance = float(A.CONFIG["STARTING_BALANCE"])
        self.last_balance_reset_trading_day = trading_day
        self.save_state()
        try:
            A.send_ntfy_notification(
                "GERCEK ISLEM | Gunluk Balance Reset",
                (
                    f"Is gunu: {trading_day} (TR cutoff 03:00)\n"
                    f"Yeni kasa: ${self.balance:.2f}\n"
                    f"Acik Gercek: {len(getattr(self, 'active_positions', {}))} | "
                    f"Acik Sanal: {len(getattr(self, 'paper_positions', {}))}"
                ),
                tags="moneybag,arrows_counterclockwise",
                priority="3",
            )
        except Exception as e:
            A.log_error("daily_balance_reset_notify", e, trading_day)
        print(f"[{now.strftime('%H:%M:%S')}] GUNLUK BALANCE RESET -> {trading_day} | ${self.balance:.2f}", flush=True)
        return True


def rotate_logs():
    import app as A

    for file_key in ["ALL_SIGNALS", "LOG", "MARKET_CONTEXT", "ERROR_LOG"]:
        path = A.FILES[file_key]
        try:
            if A.os.path.exists(path) and A.os.path.getsize(path) > A.CONFIG["MAX_LOG_SIZE_BYTES"]:
                A.os.rename(path, path + f"_old_{int(A.time.time())}")
        except Exception as e:
            A.log_error("rotate_logs", e, file_key)

