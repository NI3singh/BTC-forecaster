"""Tests for the Kronos forecaster's PURE core — path aggregation, scoring, render,
and input prep. None of this needs ``torch``/the model: ``aggregate_paths`` and the
eval scoring are numpy-only, so they run in the normal unit suite. The torch-backed
``predict()`` is exercised by the manual live smoke, not here.
"""

import numpy as np
import pandas as pd
import pytest

from tradingagents.forecasting.kronos.forecaster import (
    _auc,
    _build_inputs,
    _horizon_bars,
    _score_horizon,
    aggregate_paths,
    kronos_eval_markdown,
    render_kronos_block,
)

HORIZONS = _horizon_bars()  # [("5m",1),("15m",3),("30m",6),("1h",12),("2h",24),("4h",48)]


@pytest.mark.unit
class TestAggregatePaths:
    def test_prob_up_is_fraction_above_spot(self):
        spot = 100.0
        paths = np.full((10, 48, 6), 100.0)   # S=10, pred_len=48, 6 feats; close = idx 3
        paths[:7, 11, 3] = 101.0              # 7/10 up at 1h (bar 12 -> index 11)
        paths[7:, 11, 3] = 99.0
        out = aggregate_paths(paths, spot, HORIZONS)
        assert out["1h"] == {"prob_up": 0.7, "direction": "Up", "confidence": 70}

    def test_all_up_all_down(self):
        spot = 100.0
        paths = np.full((8, 48, 6), 100.0)
        paths[:, 47, 3] = 105.0               # all up at 4h (bar 48)
        paths[:, 0, 3] = 95.0                 # all down at 5m (bar 1)
        out = aggregate_paths(paths, spot, HORIZONS)
        assert out["4h"] == {"prob_up": 1.0, "direction": "Up", "confidence": 100}
        assert out["5m"] == {"prob_up": 0.0, "direction": "Down", "confidence": 100}

    def test_flat_band(self):
        spot = 100.0
        paths = np.full((10, 48, 6), 100.0)
        paths[:5, 11, 3] = 101.0              # 5/10 up -> prob_up 0.5 -> Flat
        paths[5:, 11, 3] = 99.0
        out = aggregate_paths(paths, spot, HORIZONS)
        assert out["1h"]["prob_up"] == 0.5
        assert out["1h"]["direction"] == "Flat"

    def test_empty_or_bad_shape_returns_empty(self):
        assert aggregate_paths(np.zeros((0, 48, 6)), 100.0, HORIZONS) == {}
        assert aggregate_paths(np.zeros((5, 6)), 100.0, HORIZONS) == {}  # not 3-D


@pytest.mark.unit
class TestBuildInputs:
    def test_columns_timestamps_spot(self):
        n = 600
        dates = pd.date_range("2026-06-01", periods=n, freq="5min")
        df = pd.DataFrame({
            "Date": dates,
            "Open": np.linspace(100, 110, n), "High": np.linspace(101, 111, n),
            "Low": np.linspace(99, 109, n), "Close": np.linspace(100, 110, n),
            "Volume": np.ones(n),
        })
        x_df, x_ts, y_ts, spot = _build_inputs(df.tail(512))
        assert list(x_df.columns) == ["open", "high", "low", "close", "volume"]
        assert len(x_df) == 512
        assert len(y_ts) == 48                         # = the longest horizon's bar index
        assert spot == float(df["Close"].iloc[-1])
        assert y_ts.iloc[0] == x_ts.iloc[-1] + pd.Timedelta(minutes=5)  # continues at 5m


@pytest.mark.unit
class TestScoring:
    def test_auc_perfect_and_one_class(self):
        assert _auc(np.array([0, 0, 1, 1.0]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
        assert np.isnan(_auc(np.ones(4), np.array([0.5, 0.6, 0.7, 0.8])))  # one class -> NaN

    def test_score_horizon_edge_vs_baselines(self):
        closes = np.full(20, 100.0)  # flat -> persistence predicts all-Down
        row = {
            "y_true": [1, 1, 0, 1, 0, 1, 1, 0, 1, 0.0],  # 6 up / 4 down
            "prob_up": [0.9, 0.8, 0.2, 0.7, 0.1, 0.6, 0.95, 0.3, 0.85, 0.05],
            "anchor": list(range(5, 15)),
        }
        st = _score_horizon(row, closes)
        assert st["n_test"] == 10
        assert st["model_acc"] == 1.0                       # threshold 0.5 separates perfectly
        assert st["best_baseline_acc"] == 0.6               # always_up / majority both 0.6
        assert round(st["edge"], 6) == round(1.0 - 0.6, 6)
        assert set(st["baselines"]) == {"always_up", "majority", "persistence"}

    def test_score_horizon_empty(self):
        assert _score_horizon({"y_true": [], "prob_up": [], "anchor": []}, np.array([])) == {"n_test": 0}


@pytest.mark.unit
class TestRender:
    def test_render_kronos_block(self):
        md = render_kronos_block("BTC-USD", {"1h": {"prob_up": 0.62, "direction": "Up", "confidence": 62}})
        assert "KRONOS MODEL" in md and "BTC-USD" in md
        assert "1h: P(up)=0.62" in md
        assert render_kronos_block("BTC-USD", {}) == ""

    def test_kronos_eval_markdown(self):
        results = {"1h": {"n_test": 100, "model_acc": 0.55, "best_baseline_acc": 0.52,
                          "best_baseline": "persistence", "edge": 0.03, "model_auc": 0.561}}
        md = kronos_eval_markdown("BTC-USD", results)
        assert "Kronos (zero-shot)" in md
        assert "| 1h | 100 |" in md
        assert "+3.0pp" in md
