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
    """Return (X, y) for one horizon, with warm-up / tail NaN rows removed."""
    data = make_features(df).join(make_label(df, horizon_bars)).dropna()
    feature_cols = [c for c in data.columns if c != "label"]
    return data[feature_cols], data["label"]
