"""Meta-labeling eval for the quant brain — honest *selective* forecasting.

Lever #1 showed the primary GBM's directional edge is weak. Meta-labeling (Lopez
de Prado) doesn't chase more accuracy; it asks "when the model commits, is it
right *there*, net of fees?" — trading recall for precision. This module measures
that with triple-barrier outcomes:

  * walk-forward the primary once (purged/embargoed) to get an honest out-of-fold
    P(up) per bar, plus the bar's triple-barrier outcome (signed return + label);
  * sweep an abstain threshold and report, per policy, COVERAGE, COMMITTED
    PRECISION (right-side rate on acted bars, time-out-flats excluded) with a
    Wilson CI, and NET-FEE PnL per trade;
  * report Brier raw vs cross-fitted-isotonic-calibrated, so the calibration gain
    is visible without the map ever training on its own output.

Phase A (here) needs no second model — calibration + abstention capture most of
the benefit. A meta-MODEL (a second GBM predicting P(primary correct)) is added
only if this eval shows a gate worth it. Offline only; never runs in live predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.agents.schemas import FORECAST_HORIZONS
from tradingagents.forecasting.quant.features import build_dataset
from tradingagents.forecasting.quant.model import generate_oos_predictions
from tradingagents.forecasting.quant.triple_barrier import triple_barrier_labels

# Abstain margins to compute (act when |P(up) - 0.5| >= margin). 0.0 = trade-all.
_MARGINS = (0.0, 0.03, 0.05, 0.08, 0.10)
# Rows actually rendered in the markdown (trade-all + one moderate + one strict).
_RENDER_MARGINS = (0.0, 0.05, 0.10)


def _calibrate_cross(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Cross-fitted isotonic calibration (2 time-halves), reusing the PAV fitter.

    Fitting and scoring on the same rows would flatter calibration; instead fit on
    one half and apply to the other (and vice versa), so every calibrated prob is
    out-of-sample for the map. Identity when there's too little data.
    """
    from tradingagents.forecasting.calibration import _fit_support, _interp
    n = len(prob)
    if n < 8:
        return prob.copy()
    mid = n // 2
    cal = np.empty(n, dtype=float)
    for (a0, a1), (b0, b1) in (((0, mid), (mid, n)), ((mid, n), (0, mid))):
        support = _fit_support(list(zip(prob[a0:a1], y[a0:a1], strict=True)))
        cal[b0:b1] = [_interp(support, float(p)) for p in prob[b0:b1]]
    return cal


def oof_with_outcomes(df: pd.DataFrame, horizon_bars: int, n_splits: int = 5,
                      pt: float = 2.0, sl: float = 2.0, sigma_window: int = 48,
                      base_deadband: float = 0.001) -> pd.DataFrame:
    """Out-of-fold primary P(up) joined to each bar's triple-barrier outcome.

    Columns: [prob_up, y_true, tb_label, tb_ret]. The walk-forward is embargoed by
    ``horizon_bars`` so the primary's OOF probs are leakage-free.
    """
    X, y = build_dataset(df, horizon_bars)
    oof = generate_oos_predictions(X, y, n_splits=n_splits, embargo=horizon_bars)
    if oof.empty:
        return oof
    tb = triple_barrier_labels(df, horizon_bars, pt=pt, sl=sl,
                               sigma_window=sigma_window, base_deadband=base_deadband)
    oof = oof.join(tb[["tb_label", "tb_ret"]])
    return oof.dropna(subset=["tb_label", "tb_ret"])


def sweep_thresholds(oof: pd.DataFrame, fee: float,
                     margins: tuple[float, ...] = _MARGINS) -> list[dict]:
    """Per abstain margin: coverage, committed precision (+Wilson CI), net-fee PnL."""
    from tradingagents.forecasting.track_record import wilson_interval
    prob = oof["prob_up"].to_numpy()
    tb_label = oof["tb_label"].to_numpy()
    tb_ret = oof["tb_ret"].to_numpy()
    n = len(prob)
    bet = np.sign(prob - 0.5)
    rows = []
    for m in margins:
        acted = (np.abs(prob - 0.5) >= m) & (bet != 0)
        n_acted = int(acted.sum())
        decided = acted & (tb_label != 0)          # exclude flat (time-out) outcomes
        n_dec = int(decided.sum())
        correct = int((bet[decided] == np.sign(tb_label[decided])).sum())
        precision = correct / n_dec if n_dec else float("nan")
        ci = wilson_interval(correct, n_dec) if n_dec else (float("nan"), float("nan"))
        trade_ret = bet[acted] * tb_ret[acted] - fee   # net-fee log-return per trade
        rows.append({
            "margin": m,
            "policy": "trade-all" if m == 0 else f"abstain >={0.5 + m:.2f}",
            "coverage": n_acted / n if n else 0.0,
            "n_acted": n_acted,
            "n_decided": n_dec,
            "precision": precision,
            "precision_ci": ci,
            "pnl_mean": float(trade_ret.mean()) if n_acted else float("nan"),
            "pnl_net": float(trade_ret.sum()),
        })
    return rows


def evaluate_meta_horizon(df: pd.DataFrame, horizon_bars: int, n_splits: int = 5,
                          fee: float = 0.001, pt: float = 2.0, sl: float = 2.0,
                          sigma_window: int = 48, base_deadband: float = 0.001,
                          margins: tuple[float, ...] = _MARGINS) -> dict:
    """Selective-forecasting metrics for one horizon (offline, walk-forward)."""
    oof = oof_with_outcomes(df, horizon_bars, n_splits=n_splits, pt=pt, sl=sl,
                            sigma_window=sigma_window, base_deadband=base_deadband)
    if oof.empty:
        return {"n": 0}
    prob = oof["prob_up"].to_numpy()
    yb = oof["y_true"].to_numpy()
    cal = _calibrate_cross(prob, yb)
    return {
        "n": len(oof),
        "fee": fee,
        "up_rate": float(yb.mean()),
        "flat_rate": float((oof["tb_label"].to_numpy() == 0).mean()),
        "brier_raw": float(np.mean((prob - yb) ** 2)),
        "brier_cal": float(np.mean((cal - yb) ** 2)),
        "sweep": sweep_thresholds(oof, fee, margins),
    }


def meta_eval_markdown(asset: str, results: dict[str, dict]) -> str:
    """Render the selective-forecasting sweep as a markdown table."""
    fee = next((r["fee"] for r in results.values() if r.get("n")), 0.001)
    lines = [
        f"## Quant meta-eval — selective forecasting ({asset}, fee {fee * 100:.2f}%/trade)",
        "",
        "_Triple-barrier outcomes. **Committed precision** = right-side rate on the "
        "bars a policy acts on (time-out flats excluded); **PnL/trade** = mean "
        "net-fee log-return per acted bar (bps). Keep a gate only where it lifts "
        "precision AND PnL with the CI clear of 50%. Brier: raw → cross-fitted "
        "isotonic (lower = better; 0.25 = a coin)._",
        "",
        "| Horizon | Policy | Coverage | Committed prec. (95% CI) | PnL/trade | Trades | Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _ in FORECAST_HORIZONS:
        st = results.get(label, {})
        if not st.get("n"):
            lines.append(f"| {label} | n/a | — | — | — | — | — |")
            continue
        by_margin = {r["margin"]: r for r in st["sweep"]}
        brier = f"{st['brier_raw']:.3f}→{st['brier_cal']:.3f}"
        first = True
        for m in _RENDER_MARGINS:
            r = by_margin.get(m)
            if r is None:
                continue
            if np.isnan(r["precision"]):
                prec = "n/a"
            else:
                lo, hi = r["precision_ci"]
                prec = f"{r['precision'] * 100:.1f}% [{lo * 100:.0f}-{hi * 100:.0f}%]"
            pnl = "n/a" if np.isnan(r["pnl_mean"]) else f"{r['pnl_mean'] * 1e4:+.1f} bps"
            lines.append(
                f"| {label if first else ''} | {r['policy']} | {r['coverage'] * 100:.0f}% | "
                f"{prec} | {pnl} | {r['n_acted']} | {brier if first else ''} |"
            )
            first = False
    return "\n".join(lines)
