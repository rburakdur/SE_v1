from __future__ import annotations

from datetime import UTC, datetime, timedelta

from rbdcrypt.models.position import ActivePosition, PositionSide


def test_position_upsert_preserves_original_opened_at_on_recovery(repos) -> None:
    t0 = datetime(2026, 2, 20, 12, 0, tzinfo=UTC)
    pos = ActivePosition(
        position_id="p1",
        symbol="BTCUSDT",
        side=PositionSide.LONG,
        qty=1.0,
        entry_price=100.0,
        initial_sl=98.0,
        initial_tp=104.0,
        current_sl=98.0,
        current_tp=104.0,
        opened_at=t0,
        recovered_at=None,
        entry_bar_time=t0,
        last_update_at=t0,
        best_pnl_pct=0.0,
        current_pnl_pct=0.0,
        leverage=1.0,
        notional=100.0,
    )
    repos.positions.upsert_active(pos)

    # Simulate buggy recovery object trying to overwrite opened_at.
    recovered = pos.model_copy(deep=True)
    recovered.opened_at = t0 + timedelta(hours=4)
    recovered.recovered_at = t0 + timedelta(hours=4)
    recovered.last_update_at = t0 + timedelta(hours=4)
    repos.positions.upsert_active(recovered)

    stored = repos.positions.get_active("p1")
    assert stored is not None
    assert stored.opened_at == t0
    assert stored.recovered_at == t0 + timedelta(hours=4)
