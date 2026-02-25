def get_signal_thresholds(config: dict, layer: str = "auto") -> dict:
    if layer == "candidate":
        return {
            "rsi_long": float(config["CANDIDATE_RSI_LONG"]),
            "rsi_short": float(config["CANDIDATE_RSI_SHORT"]),
            "vol_filter": float(config["CANDIDATE_VOL_FILTER"]),
            "adx_threshold": float(config["CANDIDATE_ADX_THRESHOLD"]),
            "min_atr_pct": float(config["CANDIDATE_MIN_ATR_PERCENT"]),
            "min_power_score": float(config["CANDIDATE_MIN_POWER_SCORE"]),
        }
    return {
        "rsi_long": float(config["AUTO_ENTRY_RSI_LONG"]),
        "rsi_short": float(config["AUTO_ENTRY_RSI_SHORT"]),
        "vol_filter": float(config["AUTO_ENTRY_VOL_FILTER"]),
        "adx_threshold": float(config["AUTO_ENTRY_ADX_THRESHOLD"]),
        "min_atr_pct": float(config["AUTO_ENTRY_MIN_ATR_PERCENT"]),
        "min_power_score": float(config["AUTO_ENTRY_MIN_POWER_SCORE"]),
    }


def get_flip_candidate_signal(row):
    flip_long = bool(row["FLIP_LONG"])
    flip_short = bool(row["FLIP_SHORT"])
    if flip_long and not flip_short:
        return "LONG"
    if flip_short and not flip_long:
        return "SHORT"
    if flip_long and flip_short:
        return "LONG" if int(row.get("TREND", 0)) >= 0 else "SHORT"
    return None


def evaluate_signal_filters(row, signal_type: str, thresholds: dict) -> dict:
    if signal_type not in ("LONG", "SHORT"):
        return {
            "flip_ok": False,
            "rsi_ok": False,
            "vol_ok": False,
            "adx_ok": False,
            "atr_ok": False,
            "ema_ok": False,
            "all_ok": False,
        }
    is_long = signal_type == "LONG"
    flip_ok = bool(row["FLIP_LONG"]) if is_long else bool(row["FLIP_SHORT"])
    rsi_ok = float(row["RSI"]) > thresholds["rsi_long"] if is_long else float(row["RSI"]) < thresholds["rsi_short"]
    vol_ok = float(row["VOL_RATIO"]) > thresholds["vol_filter"]
    adx_ok = float(row["ADX"]) > thresholds["adx_threshold"]
    atr_ok = (float(row["ATR_14"]) / max(float(row["close"]), 1e-10) * 100) >= thresholds["min_atr_pct"]
    ema_ok = float(row["close"]) > float(row["EMA20"]) if is_long else float(row["close"]) < float(row["EMA20"])
    all_ok = all([flip_ok, rsi_ok, vol_ok, adx_ok, atr_ok, ema_ok])
    return {
        "flip_ok": flip_ok,
        "rsi_ok": bool(rsi_ok),
        "vol_ok": bool(vol_ok),
        "adx_ok": bool(adx_ok),
        "atr_ok": bool(atr_ok),
        "ema_ok": bool(ema_ok),
        "all_ok": bool(all_ok),
    }


def score_from_flags(flags: dict) -> int:
    return int(sum(1 for k in ["flip_ok", "rsi_ok", "vol_ok", "adx_ok", "atr_ok", "ema_ok"] if flags.get(k)))


def hesapla_signal_score(row, *, config: dict, signal_type: str = None, thresholds: dict = None) -> int:
    thresholds = thresholds or get_signal_thresholds(config, "auto")
    if signal_type is None:
        signal_type = get_flip_candidate_signal(row)
    return score_from_flags(evaluate_signal_filters(row, signal_type, thresholds))


def hesapla_power_score(row, *, config: dict, thresholds: dict = None) -> float:
    score = 0.0
    thresholds = thresholds or get_signal_thresholds(config, "auto")

    if bool(row["FLIP_LONG"]):
        rsi_component = max(0, min(25, (float(row["RSI"]) - thresholds["rsi_long"]) * 2.5))
    else:
        rsi_component = max(0, min(25, (thresholds["rsi_short"] - float(row["RSI"])) * 2.5))
    score += rsi_component

    vol_component = max(0, min(25, (float(row["VOL_RATIO"]) - thresholds["vol_filter"]) * 15))
    score += vol_component

    adx_component = max(0, min(20, (float(row["ADX"]) - thresholds["adx_threshold"]) * 0.8))
    score += adx_component

    atr_component = max(0, min(15, (float(row["ATR_PCT"]) - thresholds["min_atr_pct"]) * 5))
    score += atr_component

    if bool(row["FLIP_LONG"]) and float(row["MACD_HIST"]) > 0:
        score += 10
    elif bool(row["FLIP_SHORT"]) and float(row["MACD_HIST"]) < 0:
        score += 10

    bb_width = (float(row["BBANDS_UP"]) - float(row["BBANDS_LOW"])) / max(float(row["BBANDS_MID"]), 1e-10) * 100
    score += max(0, min(5, bb_width * 0.5))
    return round(score, 2)


def get_candidate_fail_reason(row, candidate_signal: str, *, config: dict) -> str:
    t = get_signal_thresholds(config, "candidate")
    is_long = candidate_signal == "LONG"
    atr_pct = (float(row["ATR_14"]) / max(float(row["close"]), 1e-10)) * 100
    if is_long and not (float(row["RSI"]) > t["rsi_long"]):
        return "CAND_FAIL_RSI"
    if (not is_long) and not (float(row["RSI"]) < t["rsi_short"]):
        return "CAND_FAIL_RSI"
    if not (float(row["VOL_RATIO"]) > t["vol_filter"]):
        return "CAND_FAIL_VOL"
    if is_long and not (float(row["close"]) > float(row["EMA20"])):
        return "CAND_FAIL_EMA"
    if (not is_long) and not (float(row["close"]) < float(row["EMA20"])):
        return "CAND_FAIL_EMA"
    if not (float(row["ADX"]) > t["adx_threshold"]):
        return "CAND_FAIL_ADX"
    if not (atr_pct >= t["min_atr_pct"]):
        return "CAND_FAIL_ATR"
    return "CAND_FAIL_OTHER"


def sinyal_kontrol(row, *, config: dict):
    auto_t = get_signal_thresholds(config, "auto")
    for sig in ("LONG", "SHORT"):
        if evaluate_signal_filters(row, sig, auto_t)["all_ok"]:
            return sig
    return None

