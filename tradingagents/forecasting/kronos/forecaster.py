"""KronosForecaster: a zero-shot per-horizon directional prior from Kronos.

Kronos is a generative foundation model for OHLCV "K-lines". Used zero-shot (no
fine-tuning) it samples ``S`` future price paths; the distribution of the close at
each horizon becomes ``{prob_up, direction, confidence}`` — the SAME shape the GBM
``QuantForecaster`` produces, so this plugs straight into the existing fusion path.

``predict`` is the live entry point; ``aggregate_paths`` is the pure (numpy-only)
core that turns sampled paths into per-horizon priors (unit-tested without torch);
``evaluate`` is the offline walk-forward kill-switch that says, honestly, where
Kronos's zero-shot signal beats the naive baselines and where it doesn't.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from tradingagents.agents.schemas import FORECAST_HORIZONS

# Feature order returned by KronosPredictor.predict_samples: [open, high, low, close, volume, amount].
_CLOSE_IDX = 3
# P(up) inside 0.5 +/- this band reads as "Flat" (mirrors the GBM quant brain).
_FLAT_BAND = 0.015
_BASE_BAR_MINUTES = 5


def _direction(prob_up: float) -> str:
    if prob_up > 0.5 + _FLAT_BAND:
        return "Up"
    if prob_up < 0.5 - _FLAT_BAND:
        return "Down"
    return "Flat"


def aggregate_paths(paths, spot: float, horizons) -> dict[str, dict]:
    """Turn sampled paths ``(S, pred_len, n_features)`` into per-horizon priors.

    ``horizons`` is ``[(label, bar_index), ...]`` (e.g. ``("1h", 12)``). For each
    horizon, ``prob_up`` is the fraction of the ``S`` paths whose close at that step
    is above ``spot``; ``direction``/``confidence`` follow the quant's convention.
    Pure numpy — no torch — so it is unit-testable with synthetic paths.
    """
    arr = np.asarray(paths, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return {}
    pred_len = arr.shape[1]
    out: dict[str, dict] = {}
    for label, bar in horizons:
        if bar < 1 or bar > pred_len:
            continue
        closes = arr[:, bar - 1, _CLOSE_IDX]
        prob_up = float(np.mean(closes > spot))
        out[label] = {
            "prob_up": prob_up,
            "direction": _direction(prob_up),
            "confidence": round(100 * max(prob_up, 1 - prob_up)),
        }
    return out


def _horizon_bars() -> list[tuple[str, int]]:
    """(label, bar index in the predicted 5m path) for each forecast horizon."""
    return [(label, minutes // _BASE_BAR_MINUTES) for label, minutes in FORECAST_HORIZONS]


def _build_inputs(window: pd.DataFrame):
    """From a 5m OHLCV slice (``load_ohlcv`` shape) build ``(x_df, x_ts, y_ts, spot)``."""
    ts = pd.to_datetime(window["Date"]).reset_index(drop=True)
    x_df = pd.DataFrame({
        "open": window["Open"].to_numpy(dtype=float),
        "high": window["High"].to_numpy(dtype=float),
        "low": window["Low"].to_numpy(dtype=float),
        "close": window["Close"].to_numpy(dtype=float),
        "volume": window["Volume"].to_numpy(dtype=float),
    })
    pred_len = max(bar for _, bar in _horizon_bars())
    last = ts.iloc[-1]
    y_ts = pd.Series(pd.date_range(last + pd.Timedelta(minutes=5), periods=pred_len, freq="5min"))
    spot = float(window["Close"].iloc[-1])
    return x_df, ts, y_ts, spot


class KronosForecaster:
    """Zero-shot Kronos directional prior, one entry per forecast horizon."""

    def __init__(self, asset: str):
        from tradingagents.dataflows.config import get_config
        cfg = get_config()
        self.asset = asset
        self.model_id = cfg.get("kronos_model", "NeoQuasar/Kronos-small")
        self.tokenizer_id = cfg.get("kronos_tokenizer", "NeoQuasar/Kronos-Tokenizer-base")
        self.lookback = int(cfg.get("kronos_lookback", 512))
        self.samples = int(cfg.get("kronos_samples", 30))
        self.temperature = float(cfg.get("kronos_T", 1.0))
        self.top_p = float(cfg.get("kronos_top_p", 0.9))
        self.device = cfg.get("kronos_device", "cpu")
        self.horizons = _horizon_bars()
        self.pred_len = max(bar for _, bar in self.horizons)

    def _predictor(self):
        from tradingagents.forecasting.kronos.loader import get_predictor
        return get_predictor(self.model_id, self.tokenizer_id, self.device, self.lookback)

    def _sample(self, predictor, window: pd.DataFrame):
        x_df, x_ts, y_ts, spot = _build_inputs(window)
        paths = predictor.predict_samples(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts, pred_len=self.pred_len,
            T=self.temperature, top_p=self.top_p, sample_count=self.samples,
        )
        return paths, spot

    def predict(self) -> dict[str, dict]:
        """Per-horizon zero-shot prior; ``{label: {prob_up, direction, confidence}}`` ({} on failure)."""
        try:
            from tradingagents.dataflows.stockstats_utils import load_ohlcv
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            df = load_ohlcv(self.asset, today)
            if df is None or df.empty or len(df) < 2:
                return {}
            window = df.tail(self.lookback)
            paths, spot = self._sample(self._predictor(), window)
            return aggregate_paths(paths, spot, self.horizons)
        except Exception:
            return {}

    def evaluate(self, n_points: int = 200, stride: int = 12, log=print) -> dict[str, dict]:
        """Zero-shot walk-forward over historical 5m anchors, scored per horizon.

        At each anchor we feed Kronos the prior ``lookback`` bars and compare its
        directional call to the realized move, against the same naive baselines a
        real edge must beat. HEAVY: one model run per anchor (use ``stride`` to
        subsample). Returns the same dict shape as the quant's ``evaluate_horizon``.
        """
        from tradingagents.dataflows.stockstats_utils import load_ohlcv
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        df = load_ohlcv(self.asset, today).reset_index(drop=True)
        closes = df["Close"].to_numpy(dtype=float)
        n = len(df)
        last_anchor = n - self.pred_len - 1
        anchors = list(range(self.lookback, last_anchor, max(1, stride)))
        if not anchors:
            return {label: {"n_test": 0} for label, _ in self.horizons}
        anchors = anchors[-n_points:]
        predictor = self._predictor()

        rows = {label: {"prob_up": [], "y_true": [], "anchor": []} for label, _ in self.horizons}
        for i, a in enumerate(anchors):
            window = df.iloc[a - self.lookback + 1: a + 1]
            paths, spot = self._sample(predictor, window)
            probs = aggregate_paths(paths, spot, self.horizons)
            for label, bar in self.horizons:
                p = probs.get(label)
                if p is None:
                    continue
                rows[label]["prob_up"].append(p["prob_up"])
                rows[label]["y_true"].append(1.0 if closes[a + bar] > spot else 0.0)
                rows[label]["anchor"].append(a)
            if (i + 1) % 25 == 0 or i + 1 == len(anchors):
                log(f"  kronos-eval: {i + 1}/{len(anchors)} anchors")

        return {label: _score_horizon(rows[label], closes) for label, _ in self.horizons}


# --- Scoring (self-contained: same baseline DEFINITIONS as the GBM quant) --------

def _auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney); NaN if the slice is one-class."""
    pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _score_horizon(row: dict, closes: np.ndarray) -> dict:
    y_true = np.asarray(row["y_true"], dtype=float)
    y_prob = np.asarray(row["prob_up"], dtype=float)
    if y_true.size == 0:
        return {"n_test": 0}
    y_pred = (y_prob >= 0.5).astype(float)
    anchors = np.asarray(row["anchor"])
    # Persistence = repeat the last completed move (close[a] > close[a-1]).
    persistence = (closes[anchors] > closes[anchors - 1]).astype(float)
    # Majority = the majority class of the earliest 20% of anchors (no peeking at test).
    head = max(1, len(y_true) // 5)
    majority_cls = 1.0 if y_true[:head].mean() >= 0.5 else 0.0
    base = {
        "always_up": float((y_true == 1.0).mean()),
        "majority": float((np.full_like(y_true, majority_cls) == y_true).mean()),
        "persistence": float((persistence == y_true).mean()),
    }
    model_acc = float((y_pred == y_true).mean())
    best = max(base, key=lambda k: base[k])
    return {
        "model_acc": model_acc,
        "model_auc": _auc(y_true, y_prob),
        "baselines": base,
        "best_baseline": best,
        "best_baseline_acc": base[best],
        "edge": model_acc - base[best],
        "n_test": int(y_true.size),
        "up_rate": float(y_true.mean()),
    }


# --- Rendering -------------------------------------------------------------------

def render_kronos_block(asset: str, probs: dict[str, dict]) -> str:
    """Format Kronos per-horizon priors for injection into the PM prompt."""
    if not probs:
        return ""
    lines = [
        f"**KRONOS MODEL — generative P(up) for {asset} (zero-shot foundation model; "
        "directional prior from sampled price paths — treat as a cross-check):**",
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


def kronos_eval_markdown(asset: str, results: dict[str, dict]) -> str:
    """Render the per-horizon walk-forward evaluation as a markdown table."""
    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    lines = [
        f"## Kronos (zero-shot) — walk-forward OOS accuracy ({asset})",
        "",
        "_Per-horizon directional call from sampled Kronos paths vs the best naive "
        "baseline. Edge >0 (and AUC >0.5) = a real, measured zero-shot signal; ≈0 = no "
        "edge at that horizon (expected at the shortest ones)._",
        "",
        "| Horizon | OOS rows | Model acc | Best baseline | Edge | AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _ in FORECAST_HORIZONS:
        st = results.get(label, {})
        if not st.get("n_test"):
            lines.append(f"| {label} | 0 | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {st['n_test']} | {pct(st['model_acc'])} | "
            f"{pct(st['best_baseline_acc'])} ({st['best_baseline']}) | "
            f"{st['edge'] * 100:+.1f}pp | {st['model_auc']:.3f} |"
        )
    return "\n".join(lines)
