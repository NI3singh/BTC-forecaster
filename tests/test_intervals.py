"""Tests for the conditional volatility / prediction-interval forecaster."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from tradingagents.forecasting.quant.intervals import (
    _collect_oos,
    _metrics,
    _pinball,
    _winkler,
    evaluate_intervals,
)


def _gauss(n=4000, seed=0, vol=0.002):
    """Constant-volatility gaussian random walk — no vol clustering to exploit."""
    rng = np.random.default_rng(seed)
    logret = rng.normal(0, vol, n)
    close = 60000 * np.exp(np.cumsum(logret))
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    wick = np.abs(rng.normal(0, vol / 4, n)) * close
    return pd.DataFrame({
        "open": close, "high": close + wick, "low": close - wick, "close": close,
        "volume": np.abs(rng.normal(100, 10, n)) + 1,
        "taker_buy_base": np.abs(rng.normal(50, 5, n)),
    }, index=idx)


@pytest.mark.unit
class TestIntervalMetrics:
    def test_winkler_inside_is_width(self):
        lo, hi, y = np.array([0.0]), np.array([1.0]), np.array([0.5])
        assert _winkler(lo, hi, y, 0.8) == pytest.approx(1.0)        # width only, no miss

    def test_winkler_miss_penalty(self):
        lo, hi = np.array([0.0]), np.array([1.0])
        # alpha=0.2 -> penalty 2/alpha=10 per unit outside; 0.5 below -> 1 + 10*0.5
        assert _winkler(lo, hi, np.array([-0.5]), 0.8) == pytest.approx(6.0)
        assert _winkler(lo, hi, np.array([1.5]), 0.8) == pytest.approx(6.0)

    def test_pinball(self):
        # q=0.1, pred 0: under-predict (y=1) costs q*d=0.1; over-predict (y=-1) costs (1-q)*|d|=0.9
        assert _pinball(np.array([1.0]), np.array([0.0]), 0.1) == pytest.approx(0.1)
        assert _pinball(np.array([-1.0]), np.array([0.0]), 0.1) == pytest.approx(0.9)

    def test_metrics_coverage_and_width(self):
        lo = np.array([-1.0, -1.0, -1.0, -1.0])
        hi = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([0.0, 0.0, 2.0, 0.0])              # 3 of 4 inside
        m = _metrics(lo, hi, y, 0.8)
        assert m["coverage"] == pytest.approx(0.75)
        assert m["width"] == pytest.approx(2.0)


@pytest.mark.unit
class TestIntervalEval:
    def test_quantile_band_mostly_monotone(self):
        oos = _collect_oos(_gauss(3000, seed=2), horizon_bars=12, n_splits=4, vol_window=100)
        valid = ~np.isnan(oos["q10"]) & ~np.isnan(oos["q90"])
        assert valid.sum() > 200
        assert (oos["q90"][valid] >= oos["q10"][valid]).mean() > 0.9   # q10 <= q90

    def test_conformal_calibrates_all_methods_to_target(self):
        # The honesty check: rolling conformal must hit ~target coverage out-of-sample
        # for every method on constant-vol noise (no fake over/under-coverage).
        res = evaluate_intervals(_gauss(4000, seed=3), horizon_bars=12, n_splits=5, target=0.8)
        for name in ("baseline", "har", "quantile"):
            cal = res[name]["cal"]
            assert cal["n"] > 200
            assert abs(cal["coverage"] - 0.8) < 0.06

    def test_no_method_is_implausibly_sharp_on_noise(self):
        # With no vol clustering, a conditional method can't be dramatically sharper
        # than the baseline at equal coverage — that would betray look-ahead.
        res = evaluate_intervals(_gauss(4000, seed=5), horizon_bars=12, n_splits=5, target=0.8)
        base_w = res["baseline"]["cal"]["winkler"]
        for name in ("har", "quantile"):
            assert res[name]["cal"]["winkler"] > 0.6 * base_w


@pytest.mark.unit
class TestLiveIntervals:
    def test_predict_intervals_widen_with_horizon(self):
        from tradingagents.agents.schemas import FORECAST_HORIZONS
        from tradingagents.forecasting.quant.intervals import predict_intervals
        horizons = [(label, m // 5) for label, m in FORECAST_HORIZONS]
        off = predict_intervals(_gauss(3000, seed=4), horizons, target=0.8)
        assert len(off) == len(FORECAST_HORIZONS)
        for lo, hi in off.values():
            assert lo < 0 < hi                       # bracket the expected price
        width = {k: hi - lo for k, (lo, hi) in off.items()}
        assert width["4h"] > width["5m"]             # widen with horizon

    def test_apply_interval_ranges_overrides_only_given_horizons(self):
        from types import SimpleNamespace

        from tradingagents.agents.schemas import FORECAST_HORIZONS
        from tradingagents.forecasting.ranges import apply_interval_ranges
        fc = SimpleNamespace()
        for label, _ in FORECAST_HORIZONS:
            setattr(fc, f"expected_price_{label}", 100.0)
            setattr(fc, f"range_low_{label}", 99.0)
            setattr(fc, f"range_high_{label}", 101.0)
        apply_interval_ranges(fc, {"1h": (-0.02, 0.03)})
        assert fc.range_low_1h == pytest.approx(98.0)    # 100*(1-0.02)
        assert fc.range_high_1h == pytest.approx(103.0)  # 100*(1+0.03)
        assert fc.range_low_5m == 99.0 and fc.range_high_5m == 101.0  # untouched
