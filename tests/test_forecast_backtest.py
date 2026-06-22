"""Tests for the offline price-only baseline backtest (pure core, no network)."""

import pandas as pd
import pytest

from tradingagents.forecasting.backtest import backtest_baselines, backtest_markdown


def _bars(closes, start="2026-01-01", freq="5min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    return pd.DataFrame({"Close": list(closes)}, index=idx)


@pytest.mark.unit
class TestBacktestBaselines:
    def test_uptrend_momentum_beats_flat(self):
        # Strictly rising 5m series: every move clears the deadband upward, so the
        # realized label is always Up -> momentum (last move continues) is perfect,
        # always-Flat is useless, and the majority label is Up.
        bars = _bars([10000 + 30 * i for i in range(220)])
        res = backtest_baselines(bars, step=1)

        for h in ("5m", "1h", "4h"):
            assert res[h]["n"] > 0
            assert res[h]["momentum_acc"] == 1.0
            assert res[h]["flat_acc"] == 0.0
            assert res[h]["majority_label"] == "Up"
            assert res[h]["majority_acc"] == 1.0

    def test_constant_price_is_all_flat(self):
        # A flat series: every realized move is inside the band -> Flat everywhere.
        bars = _bars([10000.0] * 80)
        res = backtest_baselines(bars, step=1)

        assert res["1h"]["n"] > 0
        assert res["1h"]["flat_acc"] == 1.0
        assert res["1h"]["momentum_acc"] == 1.0  # prior Flat predicts realized Flat
        assert res["1h"]["majority_label"] == "Flat"

    def test_horizon_without_enough_history_reports_zero(self):
        # Only 4 bars (20 min) — the 1h/2h/4h horizons can never elapse.
        res = backtest_baselines(_bars([10000, 10050, 10100, 10150]), step=1)
        assert res["4h"]["n"] == 0
        assert res["2h"]["n"] == 0


@pytest.mark.unit
class TestBacktestMarkdown:
    def test_renders_table(self):
        bars = _bars([10000 + 30 * i for i in range(220)])
        md = backtest_markdown("BTC-USD", backtest_baselines(bars, step=6))
        assert "Baseline backtest" in md
        assert "Momentum" in md

    def test_empty_results(self):
        assert "No historical bars" in backtest_markdown("BTC-USD", {})
