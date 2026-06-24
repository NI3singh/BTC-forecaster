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


def _add_derivs(df, recent=288):
    """Add derivative columns present only for the most recent ``recent`` bars (older
    rows NaN), mimicking the ~30-day Binance futures-data window."""
    n = len(df)
    rng = np.random.default_rng(1)
    k = min(recent, n)

    def _recent(vals):
        a = np.full(n, np.nan)
        a[-k:] = vals
        return a

    out = df.copy()
    oi = np.abs(50000 + rng.normal(0, 500, k).cumsum()) + 1
    out["funding_rate"] = _recent(rng.normal(1e-4, 1e-4, k))
    out["sum_oi"] = _recent(oi)
    out["sum_oi_value"] = _recent(oi * 1e4)
    out["gls_ratio"] = _recent(np.abs(rng.normal(1.5, 0.2, k)) + 0.1)
    out["tps_ratio"] = _recent(np.abs(rng.normal(1.2, 0.15, k)) + 0.1)
    out["taker_ls_ratio"] = _recent(np.abs(rng.normal(1.0, 0.1, k)) + 0.1)
    return out


@pytest.mark.unit
class TestDerivativeFeatures:
    def test_make_features_emits_derivative_columns(self):
        feats = make_features(_add_derivs(_synth()))
        for c in ("funding_rate", "funding_cum_24", "oi_chg_12", "oi_price_div",
                  "gls_ratio", "tps_chg", "taker_ls_ratio"):
            assert c in feats.columns

    def test_build_dataset_keeps_nan_derivative_rows(self):
        base_X, _ = build_dataset(_synth(2500), horizon_bars=12)
        deriv_X, _ = build_dataset(_add_derivs(_synth(2500)), horizon_bars=12)
        assert len(deriv_X) == len(base_X)          # mostly-NaN derivs don't shrink it
        assert "oi_chg_12" in deriv_X.columns
        assert deriv_X["oi_chg_12"].isna().any()    # older rows NaN
        assert deriv_X["oi_chg_12"].notna().any()   # recent rows present

    def test_model_fits_with_nan_features(self):
        from tradingagents.forecasting.quant.model import train_full
        X, y = build_dataset(_add_derivs(_synth(2500)), horizon_bars=12)
        model = train_full(X, y)                     # HistGradientBoosting handles NaN
        assert 0.0 <= model.predict_proba(X.iloc[[-1]])[:, 1][0] <= 1.0

    def test_attach_derivatives_ffill_is_leakage_safe(self, monkeypatch):
        import tradingagents.forecasting.quant.binance_derivatives as bd
        idx = pd.date_range("2024-01-01", periods=20, freq="5min", tz="UTC")
        df = pd.DataFrame({"close": np.arange(20.0)}, index=idx)
        funding = pd.DataFrame({"funding_rate": [0.001]},
                               index=pd.DatetimeIndex(["2024-01-01 00:20:00+00:00"]))
        oi = pd.DataFrame({"sum_oi": [100.0, 110.0], "sum_oi_value": [1e6, 1.1e6]},
                          index=pd.DatetimeIndex(["2024-01-01 00:30:00+00:00",
                                                  "2024-01-01 00:45:00+00:00"]))
        empty = pd.DataFrame()
        monkeypatch.setattr(bd, "fetch_funding", lambda *a, **k: funding)
        monkeypatch.setattr(bd, "fetch_open_interest", lambda *a, **k: oi)
        monkeypatch.setattr(bd, "fetch_long_short", lambda *a, **k: empty)
        monkeypatch.setattr(bd, "fetch_taker_long_short", lambda *a, **k: empty)
        monkeypatch.setattr(bd, "_merge_cache", lambda sym, name, d: d)  # no disk write
        monkeypatch.setattr(bd, "_load_cache", lambda sym, name: None)   # no disk read
        out = bd.attach_derivatives(df, "BTCUSDT")
        fr, oi_col = out["funding_rate"].to_numpy(), out["sum_oi"].to_numpy()
        assert np.isnan(fr[3]) and fr[5] == 0.001                  # NaN before 00:20, ffilled after
        assert np.isnan(oi_col[5]) and oi_col[7] == 100.0 and oi_col[10] == 110.0


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
