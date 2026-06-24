"""QuantForecaster: per-horizon gradient-boosted P(up) for 5m..4h, plus the
walk-forward evaluation that says, honestly, where there is and isn't an edge.

This is the "quantitative brain" the agent pipeline lacked. ``predict`` returns a
calibrated directional prior per horizon; ``evaluate`` reports each horizon's
out-of-sample accuracy versus the naive baselines. Models are persisted and
retrained only when stale, so a live forecast run stays cheap after the first.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path

import joblib

from tradingagents.agents.schemas import FORECAST_HORIZONS
from tradingagents.forecasting.quant.binance_data import load_5m, to_binance_symbol
from tradingagents.forecasting.quant.features import (
    OPTIONAL_FEATURES,
    build_dataset,
    make_features,
)
from tradingagents.forecasting.quant.model import evaluate_horizon, train_full

# P(up) inside 0.5 +/- this band is reported as "Flat" (no directional lean).
_FLAT_BAND = 0.015


def _direction(prob_up: float) -> str:
    if prob_up > 0.5 + _FLAT_BAND:
        return "Up"
    if prob_up < 0.5 - _FLAT_BAND:
        return "Down"
    return "Flat"


class QuantForecaster:
    """Gradient-boosted direction models, one per forecast horizon."""

    def __init__(self, asset: str, total: int | None = None,
                 retrain_hours: float | None = None):
        from tradingagents.dataflows.config import get_config
        cfg = get_config()
        self.asset = asset
        self.symbol = to_binance_symbol(asset)
        self.total = total if total is not None else int(cfg.get("quant_total_bars", 50000))
        self.retrain_hours = (
            retrain_hours if retrain_hours is not None
            else float(cfg.get("quant_retrain_hours", 6))
        )
        # (label, horizon_in_5m_bars) for each forecast horizon.
        self.horizons = [(label, minutes // 5) for label, minutes in FORECAST_HORIZONS]

    def _model_path(self, label: str) -> Path:
        from tradingagents.dataflows.config import get_config
        d = Path(get_config().get("data_cache_dir", ".")) / "quant" / "models"
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{self.symbol}_{label}.joblib"

    def _get_model(self, df, label: str, horizon_bars: int):
        path = self._model_path(label)
        if (path.exists() and self.retrain_hours
                and time.time() - path.stat().st_mtime < self.retrain_hours * 3600):
            try:
                return joblib.load(path)
            except Exception:
                pass
        X, y = build_dataset(df, horizon_bars)
        model = train_full(X, y)
        with contextlib.suppress(Exception):
            joblib.dump(model, path)
        return model

    def predict(self, refresh: bool = False) -> dict[str, dict]:
        """Per-horizon prior for the bar after the most recent closed candle.

        Returns ``{label: {prob_up, direction, confidence}}``; ``{}`` on failure.
        """
        df = load_5m(self.asset, self.total, refresh=refresh)
        feats = make_features(df)
        core = [c for c in feats.columns if c not in OPTIONAL_FEATURES]
        feats = feats.dropna(subset=core)  # keep NaN derivative cols (the model handles them)
        if feats.empty:
            return {}
        latest = feats.iloc[[-1]]
        out: dict[str, dict] = {}
        for label, horizon_bars in self.horizons:
            model = self._get_model(df, label, horizon_bars)
            row = latest
            # Align to the columns the (possibly cached) model was trained on, so a
            # flaky derivatives endpoint can't cause a feature-name mismatch.
            if hasattr(model, "feature_names_in_"):
                row = latest.reindex(columns=list(model.feature_names_in_))
            prob_up = float(model.predict_proba(row)[:, 1][0])
            out[label] = {
                "prob_up": prob_up,
                "direction": _direction(prob_up),
                "confidence": round(100 * max(prob_up, 1 - prob_up)),
            }
        return out

    def evaluate(self, n_splits: int = 5) -> dict[str, dict]:
        """Walk-forward OOS accuracy vs baselines, per horizon (offline, honest)."""
        df = load_5m(self.asset, self.total)
        return {
            label: evaluate_horizon(*build_dataset(df, horizon_bars), df, n_splits=n_splits)
            for label, horizon_bars in self.horizons
        }


def render_quant_block(asset: str, probs: dict[str, dict]) -> str:
    """Format per-horizon quant priors for injection into the PM prompt."""
    if not probs:
        return ""
    lines = [
        f"**QUANT MODEL — gradient-boosted P(up) for {asset} (walk-forward "
        "validated prior; treat as your directional anchor, override only with "
        "strong cited evidence):**",
    ]
    for label, _ in FORECAST_HORIZONS:
        p = probs.get(label)
        if not p:
            continue
        lines.append(
            f"- {label}: P(up)={p['prob_up']:.2f} → leans {p['direction']} "
            f"(conf {p['confidence']}%)"
        )
    return "\n".join(lines)


def quant_eval_markdown(asset: str, results: dict[str, dict]) -> str:
    """Render the per-horizon walk-forward evaluation as a markdown table."""
    from tradingagents.agents.schemas import FORECAST_HORIZONS as _FH

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    lines = [
        f"## Quant model — walk-forward OOS accuracy ({asset})",
        "",
        "_Per-horizon gradient-boosted direction model vs the best naive baseline. "
        "Edge >0 (and AUC >0.5) = a real, measured directional signal; ≈0 = no edge "
        "at that horizon (expected at the shortest ones)._",
        "",
        "| Horizon | OOS rows | Model acc | Best baseline | Edge | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _ in _FH:
        st = results.get(label, {})
        if not st.get("n_test"):
            lines.append(f"| {label} | 0 | n/a | n/a | n/a | n/a |")
            continue
        edge = st["edge"]
        lines.append(
            f"| {label} | {st['n_test']} | {pct(st['model_acc'])} | "
            f"{pct(st['best_baseline_acc'])} ({st['best_baseline']}) | "
            f"{edge * 100:+.1f}pp | {st['model_auc']:.3f} |"
        )
    return "\n".join(lines)
