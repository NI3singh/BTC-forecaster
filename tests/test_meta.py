"""Tests for the meta-labeling selective-forecasting eval (synthetic, no network)."""

import numpy as np
import pandas as pd
import pytest

# Needs the [quant] extra (the eval runs the gradient-boosted primary).
pytest.importorskip("joblib")
pytest.importorskip("sklearn")

from tradingagents.forecasting.quant.meta import (
    _calibrate_cross,
    evaluate_meta_horizon,
    meta_eval_markdown,
    oof_with_outcomes,
    sweep_thresholds,
)


def _random_walk(n=4000, seed=0):
    """Pure random-walk 5m OHLCV — NO directional signal by construction.

    Increments are demeaned so there's no realized drift (else the model could
    legitimately learn the in-sample base rate and lift committed precision off
    0.5 without any look-ahead — a base-rate effect, not the leak we test for)."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(0, 120, n)
    steps -= steps.mean()                            # zero net drift
    close = 60000 + steps.cumsum()                   # ~0.2%/bar vol -> outcomes decide
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    wick = np.abs(rng.normal(0, 15, n))
    return pd.DataFrame({
        "open": close, "high": close + wick, "low": close - wick,
        "close": close, "volume": np.abs(rng.normal(100, 10, n)) + 1,
        "taker_buy_base": np.abs(rng.normal(50, 5, n)),
    }, index=idx)


@pytest.mark.unit
class TestMetaEval:
    def test_oof_with_outcomes_shape(self):
        oof = oof_with_outcomes(_random_walk(2500, seed=1), horizon_bars=12, n_splits=4)
        assert not oof.empty
        assert {"prob_up", "y_true", "tb_label", "tb_ret"}.issubset(oof.columns)
        assert set(np.unique(oof["tb_label"])).issubset({-1.0, 0.0, 1.0})

    def test_sweep_coverage_monotone(self):
        oof = oof_with_outcomes(_random_walk(2500, seed=1), horizon_bars=12, n_splits=4)
        rows = sweep_thresholds(oof, fee=0.001, margins=(0.0, 0.05, 0.10))
        covs = [r["coverage"] for r in rows]
        assert covs[0] == pytest.approx(1.0, abs=0.02)   # trade-all ~ acts on everything
        assert covs[0] >= covs[1] >= covs[2]             # stricter threshold -> less coverage

    def test_calibrate_cross_bounded(self):
        rng = np.random.default_rng(0)
        prob = rng.uniform(0, 1, 200)
        y = (rng.uniform(0, 1, 200) < prob).astype(float)
        cal = _calibrate_cross(prob, y)
        assert cal.shape == prob.shape
        assert (cal >= 0).all() and (cal <= 1).all()

    def test_no_spurious_edge_on_random_walk(self):
        # The headline safety check: an honest, leakage-free pipeline cannot
        # manufacture committed precision above chance, nor Brier below 0.25, on a
        # drift-free random walk. Asserted on the high-n trade-all policy (tight CI).
        res = evaluate_meta_horizon(_random_walk(4000, seed=7), horizon_bars=12, n_splits=5)
        assert res["n"] > 0
        trade_all = next(r for r in res["sweep"] if r["margin"] == 0.0)
        assert trade_all["n_decided"] >= 300
        assert abs(trade_all["precision"] - 0.5) < 0.05      # no look-ahead -> ~chance
        # The raw GBM is overconfident on noise (Brier can exceed 0.25); honest
        # cross-fitted calibration must pull it back to ~a coin — never BELOW, which
        # would betray a leak. This also confirms the calibration actually works.
        assert 0.243 <= res["brier_cal"] <= 0.257
        assert res["brier_cal"] <= res["brier_raw"] + 1e-9   # calibration didn't hurt

    def test_markdown_smoke(self):
        res = {
            "1h": {
                "n": 500, "fee": 0.001, "up_rate": 0.5, "flat_rate": 0.3,
                "brier_raw": 0.249, "brier_cal": 0.244,
                "sweep": [
                    {"margin": 0.0, "policy": "trade-all", "coverage": 1.0,
                     "n_acted": 500, "n_decided": 350, "precision": 0.52,
                     "precision_ci": (0.47, 0.57), "pnl_mean": -0.0001, "pnl_net": -0.05},
                    {"margin": 0.05, "policy": "abstain >=0.55", "coverage": 0.2,
                     "n_acted": 100, "n_decided": 70, "precision": 0.57,
                     "precision_ci": (0.45, 0.68), "pnl_mean": 0.0005, "pnl_net": 0.05},
                    {"margin": 0.10, "policy": "abstain >=0.60", "coverage": 0.05,
                     "n_acted": 25, "n_decided": 18, "precision": 0.60,
                     "precision_ci": (0.36, 0.80), "pnl_mean": 0.001, "pnl_net": 0.025},
                ],
            },
        }
        md = meta_eval_markdown("BTC-USD", res)
        assert "selective forecasting" in md
        assert "trade-all" in md and "abstain >=0.55" in md
        assert "0.249→0.244" in md
        assert "| 5m | n/a |" in md                      # unevaluated horizon -> n/a row
