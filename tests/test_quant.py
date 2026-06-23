"""Tests for the quant brain (synthetic data + pure helpers — no network)."""

import numpy as np
import pandas as pd
import pytest

# The quant brain needs the optional [quant] extra (scikit-learn + joblib). Skip
# the whole module cleanly when it isn't installed instead of erroring at import.
pytest.importorskip("joblib")
pytest.importorskip("sklearn")

from tradingagents.forecasting.quant.features import build_dataset, make_features, make_label
from tradingagents.forecasting.quant.forecaster import (
    _direction,
    quant_eval_markdown,
    render_quant_block,
)
from tradingagents.forecasting.quant.model import evaluate_horizon


def _synth(n=2500, seed=0):
    """Synthetic 5m OHLCV with all columns make_features needs."""
    rng = np.random.default_rng(seed)
    close = 60000 + rng.normal(0, 1, n).cumsum() * 10
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({
        "open": close + rng.normal(0, 2, n),
        "high": close + np.abs(rng.normal(0, 5, n)),
        "low": close - np.abs(rng.normal(0, 5, n)),
        "close": close,
        "volume": np.abs(rng.normal(100, 10, n)) + 1,
        "taker_buy_base": np.abs(rng.normal(50, 5, n)),
    }, index=idx)


@pytest.mark.unit
class TestFeaturesAndLabels:
    def test_label_looks_one_horizon_ahead(self):
        df = pd.DataFrame(
            {"close": [1.0, 2.0, 3.0, 2.0, 1.0]},
            index=pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC"),
        )
        label = make_label(df, horizon_bars=1)
        assert list(label[:4]) == [1.0, 1.0, 0.0, 0.0]  # up,up,down,down
        assert pd.isna(label.iloc[-1])                   # no future for last row

    def test_label_horizon_scales(self):
        df = _synth(200)
        # The last `horizon_bars` rows must be unlabeled (no future yet).
        assert make_label(df, 48).iloc[-48:].isna().all()

    def test_features_are_finite_for_recent_rows(self):
        feats = make_features(_synth()).dropna()
        assert not feats.empty
        assert np.isfinite(feats.to_numpy()).all()
        assert "taker_buy_ratio" in feats.columns   # order-flow feature present

    def test_build_dataset_aligned_and_binary(self):
        X, y = build_dataset(_synth(), horizon_bars=12)
        assert len(X) == len(y) and len(X) > 0
        assert set(y.unique()) <= {0.0, 1.0}
        assert not X.isna().any().any()


@pytest.mark.unit
class TestEvaluateHorizon:
    def test_walk_forward_returns_scored_result(self):
        X, y = build_dataset(_synth(2500, seed=1), horizon_bars=12)
        res = evaluate_horizon(X, y, _synth(2500, seed=1), n_splits=3)
        assert res["n_test"] > 0
        assert 0.0 <= res["model_acc"] <= 1.0
        assert "edge" in res and "best_baseline" in res
        assert res["edge"] == pytest.approx(res["model_acc"] - res["best_baseline_acc"])


@pytest.mark.unit
class TestDirectionMapping:
    def test_thresholds(self):
        assert _direction(0.70) == "Up"
        assert _direction(0.30) == "Down"
        assert _direction(0.50) == "Flat"
        assert _direction(0.51) == "Flat"   # inside the 0.015 flat band
        assert _direction(0.52) == "Up"


@pytest.mark.unit
class TestRendering:
    def test_quant_block(self):
        block = render_quant_block(
            "BTC-USD", {"1h": {"prob_up": 0.54, "direction": "Up", "confidence": 54}}
        )
        assert "QUANT MODEL" in block
        assert "1h: P(up)=0.54" in block
        assert render_quant_block("BTC-USD", {}) == ""

    def test_eval_markdown(self):
        md = quant_eval_markdown("BTC-USD", {
            "1h": {"n_test": 100, "model_acc": 0.54, "best_baseline_acc": 0.51,
                   "best_baseline": "always_up", "edge": 0.03, "model_auc": 0.55},
        })
        assert "Quant model" in md
        assert "+3.0pp" in md
        assert "| 5m | 0 |" in md   # unevaluated horizon renders n/a row
