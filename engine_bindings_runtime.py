def hesapla_indikatorler(df, symbol: str = ""):
    import app as A

    return A.engine_hesapla_indikatorler(df, A.CONFIG, symbol=symbol)


def hesapla_power_score(row, thresholds: dict = None) -> float:
    import app as A

    return A.engine_hesapla_power_score(row, config=A.CONFIG, thresholds=thresholds)


def hesapla_signal_score(row, signal_type: str = None, thresholds: dict = None) -> int:
    import app as A

    return A.engine_hesapla_signal_score(row, config=A.CONFIG, signal_type=signal_type, thresholds=thresholds)


def get_candidate_fail_reason(row, candidate_signal: str) -> str:
    import app as A

    return A.engine_get_candidate_fail_reason(row, candidate_signal, config=A.CONFIG)


def get_signal_thresholds(layer: str = "auto") -> dict:
    import app as A

    return A.engine_get_signal_thresholds(A.CONFIG, layer=layer)


def sinyal_kontrol(row):
    import app as A

    return A.engine_sinyal_kontrol(row, config=A.CONFIG)


def _metrics_runtime_balances() -> tuple[float, float]:
    import app as A

    try:
        return float(A.state.balance), float(A.state.peak_balance)
    except Exception:
        base = float(A.CONFIG["STARTING_BALANCE"])
        return base, base


def get_trade_performance_snapshot(trade_mode: str = "REAL") -> dict:
    import app as A

    current_balance, peak_balance = _metrics_runtime_balances()
    return A.engine_get_trade_performance_snapshot(
        A.FILES["LOG"],
        trade_mode=trade_mode,
        starting_balance=float(A.CONFIG["STARTING_BALANCE"]),
        current_balance=current_balance,
        peak_balance=peak_balance,
        log_error=A.log_error,
    )


def get_advanced_metrics(trade_mode: str = "REAL"):
    import app as A

    current_balance, peak_balance = _metrics_runtime_balances()
    return A.engine_get_advanced_metrics(
        A.FILES["LOG"],
        trade_mode=trade_mode,
        starting_balance=float(A.CONFIG["STARTING_BALANCE"]),
        current_balance=current_balance,
        peak_balance=peak_balance,
        log_error=A.log_error,
    )


def calc_real_mdd_from_trade_log(df) -> float:
    import app as A

    current_balance, peak_balance = _metrics_runtime_balances()
    return A.engine_calc_real_mdd_from_trade_log(
        df,
        starting_balance=float(A.CONFIG["STARTING_BALANCE"]),
        current_balance=current_balance,
        peak_balance=peak_balance,
        log_error=A.log_error,
    )

