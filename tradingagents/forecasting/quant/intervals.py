"""Conditional volatility / prediction-interval forecaster for the quant brain.

Direction at 5m is ~efficient (levers #1, #2 found no edge), but **volatility is
forecastable** (clustering; HAR-RV and gradient boosting beat GARCH for short-term
crypto vol). The forecast contract sells an 80% price range per horizon; today that
range is a CONSTANT-sigma band (``ranges.vol_scaled_band`` = expected*(1 +/-
1.5*sigma*sqrt(t))) — one trailing sigma, fixed k, normality-implied. This module
forecasts the 80% interval of the h-step log-return CONDITIONALLY and measures,
walk-forward, whether it beats that baseline on the honest interval score.

Everything is in log-return space, centered at 0 (random-walk center — honest given
~zero directional edge). Three methods are scored head-to-head, each also wrapped in
a rolling conformal layer that forces ~80% coverage so the comparison reduces to
SHARPNESS at equal coverage:

  * baseline  — k*sigma_trailing*sqrt(h)              (the live constant-sigma band)
  * har       — HAR-RV conditional sigma (OLS on multi-scale realized vol) * z
  * quantile  — HistGradientBoostingRegressor(loss="quantile") at q=0.1/0.9

Metrics: coverage, mean width, pinball loss, and the Winkler interval score (width
+ miss penalty — the single honest scalar; widening alone never wins).

LEAKAGE: the targets (h-step return; future realized vol) are the only
forward-looking quantities; features are backward-only and the walk-forward purges
``embargo=horizon_bars``. Conformal calibrates from PAST realized residuals only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tradingagents.forecasting.quant.features import OPTIONAL_FEATURES, make_features
from tradingagents.forecasting.quant.model import _usable_columns

_Z80 = 1.2815515594457831          # normal 90th pct -> 80% central interval half-width
_HAR_FEATURES = ["vol_12", "vol_48", "vol_144"]   # multi-scale realized vol (HAR components)
_RANGE_K = 1.5                     # the live baseline's k (forecasting.ranges.RANGE_K)

_GBM_KW = {"max_iter": 200, "learning_rate": 0.05, "max_depth": 3,
           "l2_regularization": 1.0, "early_stopping": True,
           "validation_fraction": 0.15, "random_state": 42}


def _quantile_gbm(quantile: float) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, **_GBM_KW)


def _pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss at level ``q`` — the proper score for a quantile."""
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def _winkler(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, target: float) -> float:
    """Winkler interval score for a central ``target`` interval (lower = better).

    Width plus a miss penalty of 2/alpha times how far ``y`` lands outside, so a
    method cannot win by merely widening — the honest single scalar.
    """
    alpha = 1.0 - target
    width = hi - lo
    below = (y < lo) * (lo - y)
    above = (y > hi) * (y - hi)
    return float(np.mean(width + (2.0 / alpha) * (below + above)))


def _metrics(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, target: float) -> dict:
    mask = ~np.isnan(lo) & ~np.isnan(hi)
    lo, hi, y = lo[mask], hi[mask], y[mask]
    if len(y) == 0:
        return {"n": 0}
    return {
        "n": int(len(y)),
        "coverage": float(np.mean((y >= lo) & (y <= hi))),
        "width": float(np.mean(hi - lo)),               # log-return units ~ price fraction
        "winkler": _winkler(lo, hi, y, target),
    }


def _rolling_conformal(scores: np.ndarray, target: float, window: int, min_cal: int) -> np.ndarray:
    """Per-row conformal constant = ``target`` quantile of the PRIOR ``window`` scores.

    Rolling (not global) so it adapts to regime rather than assuming exchangeability.
    NaN until ``min_cal`` past residuals exist.
    """
    n = len(scores)
    c = np.full(n, np.nan)
    for t in range(n):
        cal = scores[max(0, t - window):t]
        cal = cal[~np.isnan(cal)]
        if len(cal) >= min_cal:
            c[t] = np.quantile(cal, target)
    return c


def _collect_oos(df: pd.DataFrame, horizon_bars: int, n_splits: int,
                 vol_window: int) -> dict:
    """Walk-forward (expanding + embargo) raw per-OOS-row predictions for each method."""
    feats = make_features(df)
    logret = np.log(df["close"] / df["close"].shift(1))
    sigma_bar = logret.rolling(vol_window).std()
    r_h = np.log(df["close"].shift(-horizon_bars) / df["close"])
    fut_vol = np.sqrt((logret ** 2).rolling(horizon_bars).mean().shift(-horizon_bars))

    core = [c for c in feats.columns if c not in OPTIONAL_FEATURES]
    data = (feats.join(r_h.rename("r_h")).join(sigma_bar.rename("sig"))
            .join(fut_vol.rename("fvol")))
    data = data.dropna(subset=core + ["r_h", "sig", "fvol"])
    feat_cols = list(feats.columns)
    X = data[feat_cols]
    y = data["r_h"].to_numpy()
    sig = data["sig"].to_numpy()
    fvol = data["fvol"].to_numpy()

    n = len(X)
    fold = n // (n_splits + 1)
    emb = horizon_bars
    sqrt_h = np.sqrt(horizon_bars)
    out = {k: np.full(n, np.nan) for k in ("y", "base_hw", "har_hw", "q10", "q90")}
    out["y"] = y
    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_end = fold * (k + 1) if k < n_splits else n
        tr_hi = max(0, train_end - emb)
        if tr_hi <= 0 or test_end <= train_end:
            continue
        cols = _usable_columns(X.iloc[:tr_hi])
        Xtr = X.iloc[:tr_hi][cols]
        Xte = X.iloc[train_end:test_end][cols]
        te = slice(train_end, test_end)

        # Baseline: live constant-sigma band (trailing per-bar vol * sqrt(t) * k).
        out["base_hw"][te] = _RANGE_K * sig[te] * sqrt_h
        # HAR-RV: OLS of next-horizon realized per-bar vol on multi-scale realized vol.
        har_cols = [c for c in _HAR_FEATURES if c in cols]
        if har_cols:
            A = np.column_stack([np.ones(tr_hi), X.iloc[:tr_hi][har_cols].to_numpy()])
            coef, *_ = np.linalg.lstsq(A, fvol[:tr_hi], rcond=None)
            Ate = np.column_stack([np.ones(len(Xte)), X.iloc[train_end:test_end][har_cols].to_numpy()])
            sig_hat = np.clip(Ate @ coef, 1e-6, None)
            out["har_hw"][te] = _Z80 * sig_hat * sqrt_h
        # Quantile-GBM: direct q0.1 / q0.9 of the h-step log-return.
        out["q10"][te] = _quantile_gbm(0.1).fit(Xtr, y[:tr_hi]).predict(Xte)
        out["q90"][te] = _quantile_gbm(0.9).fit(Xtr, y[:tr_hi]).predict(Xte)
    return out


def evaluate_intervals(df: pd.DataFrame, horizon_bars: int, n_splits: int = 5,
                       target: float = 0.8, vol_window: int = 100,
                       conformal_window: int = 750, min_cal: int = 250) -> dict:
    """Per-method interval metrics for one horizon (raw + rolling-conformal)."""
    oos = _collect_oos(df, horizon_bars, n_splits, vol_window)
    y = oos["y"]
    methods = {
        "baseline": (-oos["base_hw"], oos["base_hw"]),
        "har": (-oos["har_hw"], oos["har_hw"]),
        "quantile": (oos["q10"], oos["q90"]),
    }
    result: dict[str, dict] = {"horizon_bars": horizon_bars, "target": target}
    for name, (lo, hi) in methods.items():
        valid = ~np.isnan(lo) & ~np.isnan(hi)
        if valid.sum() < min_cal:
            result[name] = {"n": 0}
            continue
        raw = _metrics(lo, hi, y, target)
        # Conformal: symmetric methods scale the half-width; quantile uses CQR (additive).
        if name == "quantile":
            scores = np.maximum(lo - y, y - hi)            # CQR nonconformity
            c = _rolling_conformal(scores, target, conformal_window, min_cal)
            clo, chi = lo - c, hi + c
            pinball = (_pinball(y[valid], lo[valid], 0.1) + _pinball(y[valid], hi[valid], 0.9)) / 2
            raw["pinball"] = pinball
        else:
            hw = hi
            scores = np.abs(y) / np.where(hw > 0, hw, np.nan)
            c = _rolling_conformal(scores, target, conformal_window, min_cal)
            chi = c * hw
            clo = -chi
        cal = _metrics(clo, chi, y, target)
        result[name] = {"raw": raw, "cal": cal}
    return result


def intervals_eval_markdown(asset: str, results: dict[str, dict]) -> str:
    """Render the per-horizon, per-method interval comparison as markdown."""
    from tradingagents.agents.schemas import FORECAST_HORIZONS
    target = next((r.get("target", 0.8) for r in results.values() if r.get("baseline")), 0.8)
    lines = [
        f"## Quant interval-eval — 80% range calibration & sharpness ({asset})",
        "",
        f"_Conditional {int(target * 100)}% interval of the h-step return vs the live "
        "constant-σ band. Each method is rolling-conformal-calibrated to the target so "
        "coverage matches and the contest is **sharpness**: lower **Width** and **Winkler** "
        "(width + miss penalty) win. A method earns 'keep' only if its conformal Winkler "
        "beats the baseline's. Widths are % of price._",
        "",
        "| Horizon | Method | Cov (cal) | Width% (cal) | Winkler (cal) | Cov (raw) |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, _ in FORECAST_HORIZONS:
        st = results.get(label, {})
        base = st.get("baseline", {}).get("cal", {}) if st else {}
        first = True
        for name in ("baseline", "har", "quantile"):
            m = st.get(name) if st else None
            if not m or not m.get("cal", {}).get("n"):
                continue
            cal, raw = m["cal"], m["raw"]
            win = cal["winkler"]
            flag = ""
            if name != "baseline" and base.get("winkler") and win < base["winkler"]:
                flag = " ✅"
            lines.append(
                f"| {label if first else ''} | {name} | {cal['coverage'] * 100:.0f}% | "
                f"{cal['width'] * 100:.2f}% | {win * 1e4:.1f}{flag} | {raw['coverage'] * 100:.0f}% |"
            )
            first = False
        if first:
            lines.append(f"| {label} | n/a | — | — | — | — |")
    return "\n".join(lines)
