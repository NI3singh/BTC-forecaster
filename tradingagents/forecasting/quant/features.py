"""Leakage-free features for 5-minute BTC direction prediction.

Adapted from trading_bot/src/features.py, with windows re-expressed in 5-minute
bars and spanning 5 minutes to ~1 day so the models have context for every
horizon from 5m to 4h.

LEAKAGE RULE: every feature for bar t uses only bars t, t-1, t-2, ...  The only
forward-looking quantity is the label (``make_label``), which is what we predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Lookbacks in 5-minute bars: 5m, 15m, 30m, 1h, 2h, 4h, 12h, 1d.
_RET_LAGS = [1, 3, 6, 12, 24, 48, 144, 288]
_VOL_WINS = [6, 12, 24, 48, 144]
_SMA_WINS = [12, 24, 48, 144, 288]

# Derivatives / order-flow features (from binance_derivatives.attach_derivatives).
# OPTIONAL: present only when the futures endpoints succeed, and NaN for bars older
# than the ~30-day OI/long-short window. HistGradientBoostingClassifier handles that
# NaN natively, so build_dataset drops rows on the CORE features only — never these.
OPTIONAL_FEATURES = {
    "funding_rate", "funding_chg", "funding_cum_24",
    "oi_chg_3", "oi_chg_12", "oi_chg_48", "oi_price_div", "oi_value_chg",
    "gls_ratio", "gls_chg", "tps_ratio", "tps_chg", "taker_ls_ratio",
}


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build a feature matrix from a 5m OHLCV frame (oldest -> newest)."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    logret = np.log(close / close.shift(1))

    # Momentum: past log returns over several horizons.
    for lag in _RET_LAGS:
        out[f"ret_{lag}"] = np.log(close / close.shift(lag))
    # Volatility: rolling std of 5m returns.
    for win in _VOL_WINS:
        out[f"vol_{win}"] = logret.rolling(win).std()
    # Trend: price relative to moving averages.
    for win in _SMA_WINS:
        out[f"close_over_sma_{win}"] = close / close.rolling(win).mean() - 1.0

    # Oscillators.
    out["rsi_14"] = _rsi(close, 14)
    out["rsi_48"] = _rsi(close, 48)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    out["macd"] = macd / close
    out["macd_signal"] = macd.ewm(span=9, adjust=False).mean() / close

    # Candle shape (current closed candle).
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    out["body_frac"] = (df["close"] - df["open"]) / rng
    out["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / rng
    out["hl_range"] = rng / close

    # Volume + order flow (taker_buy_base = share of volume that was aggressive buying).
    vol = df["volume"].replace(0, np.nan)
    out["vol_chg"] = np.log(vol / vol.shift(1))
    out["vol_over_ma48"] = vol / vol.rolling(48).mean() - 1.0
    if "taker_buy_base" in df.columns:
        out["taker_buy_ratio"] = df["taker_buy_base"] / vol

    # Derivatives / positioning (optional; present only when the futures endpoints
    # were fetched — see binance_derivatives.attach_derivatives). Use changes and
    # divergences, not raw levels (OI/ratios are non-stationary). All backward-looking.
    if "funding_rate" in df.columns:
        fr = pd.to_numeric(df["funding_rate"], errors="coerce")
        out["funding_rate"] = fr
        out["funding_chg"] = fr.diff()
        out["funding_cum_24"] = fr.rolling(24).sum()
    if "sum_oi" in df.columns:
        oi = pd.to_numeric(df["sum_oi"], errors="coerce").replace(0, np.nan)
        for win in (3, 12, 48):
            out[f"oi_chg_{win}"] = np.log(oi / oi.shift(win))
        # OI vs price divergence: +1 = price & OI move together (real move); -1 =
        # opposite (e.g. price up on falling OI = short-covering, fragile).
        out["oi_price_div"] = np.sign(np.log(oi / oi.shift(12))) * np.sign(out["ret_12"])
    if "sum_oi_value" in df.columns:
        oiv = pd.to_numeric(df["sum_oi_value"], errors="coerce").replace(0, np.nan)
        out["oi_value_chg"] = np.log(oiv / oiv.shift(12))
    if "gls_ratio" in df.columns:  # global (retail) long/short accounts — contrarian
        gls = pd.to_numeric(df["gls_ratio"], errors="coerce")
        out["gls_ratio"] = gls
        out["gls_chg"] = gls.diff()
    if "tps_ratio" in df.columns:  # top-trader long/short positions — smart money
        tps = pd.to_numeric(df["tps_ratio"], errors="coerce")
        out["tps_ratio"] = tps
        out["tps_chg"] = tps.diff()
    if "taker_ls_ratio" in df.columns:  # futures taker buy/sell aggressive flow
        out["taker_ls_ratio"] = pd.to_numeric(df["taker_ls_ratio"], errors="coerce")

    # Calendar (cyclical so 23h is next to 0h).
    hour, dow = df.index.hour, df.index.dayofweek
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out


def make_label(df: pd.DataFrame, horizon_bars: int) -> pd.Series:
    """Binary label: 1 if the close ``horizon_bars`` ahead is higher than now.

    The only forward-looking quantity. ``label[t] = close[t + horizon_bars] >
    close[t]``; the final ``horizon_bars`` rows are NaN (no future yet) and dropped.
    """
    future = df["close"].shift(-horizon_bars)
    label = (future > df["close"]).astype(float)
    label[future.isna()] = np.nan
    return label.rename("label")


def build_dataset(df: pd.DataFrame, horizon_bars: int) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for one horizon. Warm-up/tail NaN rows are dropped on the CORE
    features + label only; OPTIONAL derivative features keep their NaN (the model
    handles it) so the ~30-day futures-data window doesn't shrink the training set."""
    feats = make_features(df)
    data = feats.join(make_label(df, horizon_bars))
    core = [c for c in feats.columns if c not in OPTIONAL_FEATURES]
    data = data.dropna(subset=core + ["label"])
    feature_cols = [c for c in data.columns if c != "label"]
    return data[feature_cols], data["label"]
