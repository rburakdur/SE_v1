import csv
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd


LogErrorFn = Optional[Callable[[str, Exception, str], None]]


def _safe_log_error(log_error: LogErrorFn, context: str, error: Exception, extra: str = "") -> None:
    if not callable(log_error):
        return
    try:
        log_error(context, error, extra)
    except Exception:
        pass


def _coerce_trade_log_numeric(series: pd.Series, log_error: LogErrorFn = None) -> pd.Series:
    try:
        if series is None:
            return pd.Series(dtype="float64")
        if pd.api.types.is_numeric_dtype(series):
            return pd.to_numeric(series, errors="coerce")

        s = series.astype(str).str.strip()
        s = s.replace(
            {
                "": np.nan,
                "nan": np.nan,
                "NaN": np.nan,
                "None": np.nan,
                "NONE": np.nan,
                "<NA>": np.nan,
            }
        )
        s = s.str.replace("%", "", regex=False).str.replace("$", "", regex=False)
        s = s.str.replace(" ", "", regex=False)

        comma_decimal = s.str.contains(",", regex=False, na=False) & ~s.str.contains(".", regex=False, na=False)
        if bool(comma_decimal.any()):
            s.loc[comma_decimal] = s.loc[comma_decimal].str.replace(",", ".", regex=False)
        s = s.str.replace(",", "", regex=False)
        return pd.to_numeric(s, errors="coerce")
    except Exception as e:
        _safe_log_error(log_error, "_coerce_trade_log_numeric", e)
        return pd.to_numeric(series, errors="coerce")


def _read_hunter_log_df(log_path: str, log_error: LogErrorFn = None) -> pd.DataFrame:
    if not log_path or not os.path.exists(log_path):
        return pd.DataFrame()
    try:
        try:
            return pd.read_csv(log_path, encoding="utf-8-sig", low_memory=False, on_bad_lines="skip")
        except TypeError:
            return pd.read_csv(log_path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(log_path, low_memory=False)
    except Exception as e:
        _safe_log_error(log_error, "_read_hunter_log_df", e, str(log_path))
        return pd.DataFrame()


def _read_hunter_log_mixed_schema_metrics_df(log_path: str, log_error: LogErrorFn = None) -> pd.DataFrame:
    """
    hunter_history.csv aynı dosyada birden fazla şema ile append edildiğinde
    (örn. 42 kolon eski, 43 kolon yeni REAL, 36 kolon VIRTUAL) metrik için
    gerekli kolonları satır bazında güvenli çıkarır.
    """
    if not log_path or not os.path.exists(log_path):
        return pd.DataFrame()

    mode_tokens = {"REAL", "VIRTUAL", "PAPER", "SIM", "SANAL"}
    out_rows = []
    try:
        with open(log_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            header_map = {str(c): i for i, c in enumerate(header)}
            header_len = len(header)
            has_trade_mode_header = "Trade_Mode" in header_map

            for line_no, row in enumerate(reader, start=2):
                if not row:
                    continue
                try:
                    first = str(row[0]).strip().upper() if row else ""
                    explicit_mode = first if first in mode_tokens else None
                    rec = {"_line_no": line_no, "_row_len": len(row)}

                    if explicit_mode and len(row) >= 19:
                        # Yeni şemalar: Trade_Mode ve Restarted başta, PnL blokları sabit indekslerde.
                        rec["Trade_Mode"] = explicit_mode
                        rec["Scan_ID"] = row[1] if len(row) > 1 else None
                        rec["Tarih"] = row[3] if len(row) > 3 else None
                        rec["Coin"] = row[7] if len(row) > 7 else None
                        rec["PnL_Yuzde"] = row[15] if len(row) > 15 else None
                        rec["PnL_USD"] = row[16] if len(row) > 16 else None
                        rec["Kasa_Son_Durum"] = row[17] if len(row) > 17 else None
                        rec["Sonuc"] = row[18] if len(row) > 18 else None
                        out_rows.append(rec)
                        continue

                    if header_len and len(row) == header_len:
                        # Eski/uyumlu header satırı: header map ile oku.
                        rec["Trade_Mode"] = row[header_map["Trade_Mode"]] if has_trade_mode_header else "REAL"
                        for col in ("Scan_ID", "Tarih", "Coin", "PnL_Yuzde", "PnL_USD", "Kasa_Son_Durum", "Sonuc"):
                            idx = header_map.get(col)
                            rec[col] = row[idx] if idx is not None and idx < len(row) else None
                        out_rows.append(rec)
                        continue
                except Exception as row_err:
                    _safe_log_error(log_error, "_read_hunter_log_mixed_schema_metrics_df_row", row_err, f"line={line_no}")
                    continue
    except Exception as e:
        _safe_log_error(log_error, "_read_hunter_log_mixed_schema_metrics_df", e, str(log_path))
        return pd.DataFrame()

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows)


def _normalize_hunter_log_df(df: pd.DataFrame, log_error: LogErrorFn = None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    for col in ["PnL_Yuzde", "PnL_USD", "Kasa_Son_Durum", "Risk_USD"]:
        if col in out.columns:
            out[col] = _coerce_trade_log_numeric(out[col], log_error=log_error)
    if "Trade_Mode" in out.columns:
        out["Trade_Mode"] = (
            out["Trade_Mode"].astype(str).str.strip().str.upper().replace({"NAN": "", "NONE": "", "<NA>": ""})
        )
    return out


def _filter_hunter_log_by_trade_mode(df: pd.DataFrame, trade_mode: str = "REAL") -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    mode = str(trade_mode or "REAL").strip().upper()
    if "Trade_Mode" not in df.columns:
        return df.copy() if mode == "REAL" else df.iloc[0:0].copy()

    tm = df["Trade_Mode"].fillna("").astype(str).str.strip().str.upper()
    tm = tm.replace({"NAN": "", "NONE": "", "<NA>": ""})
    if mode == "REAL":
        mask = tm.isin(["", "REAL"])
    elif mode == "VIRTUAL":
        mask = tm.isin(["VIRTUAL", "PAPER", "SIM", "SANAL"])
    else:
        mask = tm == mode
    return df[mask].copy()


def _get_hunter_log_for_metrics(log_path: str, trade_mode: str = "REAL", log_error: LogErrorFn = None) -> pd.DataFrame:
    # Önce mixed-schema dayanıklı parser; boş dönerse pandas fallback.
    df = _read_hunter_log_mixed_schema_metrics_df(log_path, log_error=log_error)
    if len(df) == 0:
        df = _read_hunter_log_df(log_path, log_error=log_error)
    if len(df) == 0:
        return df
    df = _normalize_hunter_log_df(df, log_error=log_error)
    return _filter_hunter_log_by_trade_mode(df, trade_mode)


def calc_real_mdd_from_trade_log(
    df: pd.DataFrame,
    *,
    starting_balance: float,
    current_balance: Optional[float] = None,
    peak_balance: Optional[float] = None,
    log_error: LogErrorFn = None,
) -> float:
    try:
        if df is None or len(df) == 0:
            return 0.0

        if "Kasa_Son_Durum" in df.columns:
            equity = pd.to_numeric(df["Kasa_Son_Durum"], errors="coerce").dropna()
        elif "PnL_USD" in df.columns:
            pnl_usd = pd.to_numeric(df["PnL_USD"], errors="coerce").fillna(0.0)
            equity = pnl_usd.cumsum() + float(starting_balance)
        else:
            if current_balance is None or peak_balance is None:
                return 0.0
            return round(((float(peak_balance) - float(current_balance)) / max(float(peak_balance), 1e-9)) * 100, 2)

        if len(equity) == 0:
            return 0.0

        running_peak = equity.cummax()
        dd_pct = ((running_peak - equity) / running_peak.replace(0, np.nan)) * 100
        return round(float(dd_pct.fillna(0.0).max()), 2)
    except Exception as e:
        _safe_log_error(log_error, "calc_real_mdd_from_trade_log", e)
        if current_balance is None or peak_balance is None:
            return 0.0
        return round(((float(peak_balance) - float(current_balance)) / max(float(peak_balance), 1e-9)) * 100, 2)


def get_trade_performance_snapshot(
    log_path: str,
    *,
    trade_mode: str = "REAL",
    starting_balance: float = 100.0,
    current_balance: Optional[float] = None,
    peak_balance: Optional[float] = None,
    log_error: LogErrorFn = None,
) -> dict:
    snapshot = {
        "total_trades": 0,
        "wins": 0,
        "win_rate": 0,
        "pf": 0.0,
        "max_dd": 0.0,
        "net_pnl_usd": 0.0,
    }
    try:
        df = _get_hunter_log_for_metrics(log_path, trade_mode=trade_mode, log_error=log_error)
        if len(df) == 0:
            return snapshot

        pnl_basis_col = "PnL_Yuzde" if "PnL_Yuzde" in df.columns else ("PnL_USD" if "PnL_USD" in df.columns else None)
        if not pnl_basis_col:
            return snapshot

        pnl_basis = pd.to_numeric(df[pnl_basis_col], errors="coerce")
        valid_mask = pnl_basis.notna()
        if not bool(valid_mask.any()):
            return snapshot

        df = df.loc[valid_mask].copy()
        pnl_basis = pnl_basis.loc[valid_mask]

        pnl_usd = df["PnL_USD"].fillna(0.0) if "PnL_USD" in df.columns else None
        pnl_for_pf = pnl_usd if pnl_usd is not None else pnl_basis.fillna(0.0)
        win_mask = pnl_basis > 0
        loss_mask = pnl_basis <= 0

        total_trades = int(len(df))
        wins = int(win_mask.sum())
        gross_profit = float(pnl_for_pf[win_mask].sum())
        gross_loss = abs(float(pnl_for_pf[loss_mask].sum()))
        if gross_loss > 0:
            pf = round(gross_profit / gross_loss, 2)
        elif gross_profit > 0:
            pf = 99.9
        else:
            pf = 0.0

        max_dd = (
            calc_real_mdd_from_trade_log(
                df,
                starting_balance=starting_balance,
                current_balance=current_balance,
                peak_balance=peak_balance,
                log_error=log_error,
            )
            if str(trade_mode).upper() == "REAL"
            else 0.0
        )
        net_pnl_usd = float(pnl_usd.sum()) if pnl_usd is not None else 0.0

        snapshot.update(
            {
                "total_trades": total_trades,
                "wins": wins,
                "win_rate": int((wins / total_trades) * 100) if total_trades > 0 else 0,
                "pf": pf,
                "max_dd": max_dd,
                "net_pnl_usd": round(net_pnl_usd, 2),
            }
        )
        return snapshot
    except Exception as e:
        _safe_log_error(log_error, "get_trade_performance_snapshot", e, str(trade_mode))
        return snapshot


def get_advanced_metrics(
    log_path: str,
    *,
    trade_mode: str = "REAL",
    starting_balance: float = 100.0,
    current_balance: Optional[float] = None,
    peak_balance: Optional[float] = None,
    log_error: LogErrorFn = None,
) -> tuple[int, int, int, float, float]:
    snapshot = get_trade_performance_snapshot(
        log_path,
        trade_mode=trade_mode,
        starting_balance=starting_balance,
        current_balance=current_balance,
        peak_balance=peak_balance,
        log_error=log_error,
    )
    return (
        snapshot["total_trades"],
        snapshot["wins"],
        snapshot["win_rate"],
        snapshot["pf"],
        snapshot["max_dd"],
    )
