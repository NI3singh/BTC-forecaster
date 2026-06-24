"""Fetch 5-minute OHLCV for the quant models from the Binance public API.

Adapted from trading_bot/src/data.py. Binance is a strict upgrade over yfinance
for this purpose: real exchange volume, the ``taker_buy_base`` order-flow column,
and years of 5-minute history (yfinance caps 5m at ~60 days). No API key needed.
Closed candles only; cached under the forecaster's data_cache_dir.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

BINANCE_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000  # Binance max candles per request

_RAW_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades",
    "taker_buy_base", "taker_buy_quote", "ignore",
]

_INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def to_binance_symbol(asset: str) -> str:
    """Map the forecaster's ticker (e.g. ``BTC-USD``) to a Binance pair (``BTCUSDT``)."""
    base = asset.split("-")[0].split("/")[0].upper()
    return f"{base}USDT"


def _cache_path(symbol: str, interval: str) -> Path:
    from tradingagents.dataflows.config import get_config
    cache_dir = Path(get_config().get("data_cache_dir", ".")) / "quant"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{symbol}_{interval}.csv"


def _rows_to_df(rows: list[list]) -> pd.DataFrame:
    """Parse raw Binance kline rows into a typed, timestamp-indexed frame.

    Only CLOSED candles are kept (close_time strictly in the past).
    """
    df = pd.DataFrame(rows, columns=_RAW_COLS)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")
    df = df[df["close_time"].astype("int64") < time.time() * 1000]
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]:
        df[col] = df[col].astype(float)
    df["trades"] = df["trades"].astype(int)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    return df[["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]]


def fetch_klines(symbol: str, interval: str, total: int) -> pd.DataFrame:
    """Fetch the most recent ``total`` closed candles, paginating backwards."""
    all_rows: list[list] = []
    end_time: int | None = None
    while len(all_rows) < total:
        params = {"symbol": symbol, "interval": interval,
                  "limit": min(MAX_LIMIT, total - len(all_rows))}
        if end_time is not None:
            params["endTime"] = end_time
        resp = requests.get(BINANCE_URL, params=params, timeout=20)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        all_rows = batch + all_rows
        end_time = batch[0][0] - 1
        time.sleep(0.2)
        if len(batch) < params["limit"]:
            break
    return _rows_to_df(all_rows)


def _load_klines(asset: str, total: int = 50000, interval: str = "5m",
                 refresh: bool = False) -> pd.DataFrame:
    """Load closed candles, keeping the on-disk cache reasonably current.

    Full refetch when the cache is missing/too small or ``refresh``; otherwise the
    cache is reused if its newest candle is within two bars of now, else topped up.
    Returns the most recent ``total`` rows, oldest -> newest. Never raises into the
    caller on a network hiccup if a usable cache exists.
    """
    symbol = to_binance_symbol(asset)
    cache = _cache_path(symbol, interval)
    bar_ms = _INTERVAL_MS.get(interval, 300_000)

    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col="timestamp", parse_dates=["timestamp"])
        if len(df) >= total * 0.9:
            last_ms = df.index[-1].value // 1_000_000
            if time.time() * 1000 - last_ms < 2 * bar_ms:
                return df.tail(total)
            try:
                new = fetch_klines(symbol, interval, MAX_LIMIT)
                df = pd.concat([df, new])
                df = df[~df.index.duplicated(keep="last")].sort_index()
                df.to_csv(cache)
            except requests.RequestException:
                pass  # keep serving the cached frame
            return df.tail(total)

    df = fetch_klines(symbol, interval, total)
    df.to_csv(cache)
    return df


def load_5m(asset: str, total: int = 50000, interval: str = "5m",
            refresh: bool = False) -> pd.DataFrame:
    """Closed 5m candles, plus Binance Futures derivatives/order-flow columns when
    ``quant_derivatives`` is enabled (funding, OI, long/short — best-effort, never
    raises). Both ``QuantForecaster.predict`` and ``.evaluate`` route through here."""
    df = _load_klines(asset, total, interval, refresh)
    from tradingagents.dataflows.config import get_config
    if interval == "5m" and get_config().get("quant_derivatives"):
        try:
            from tradingagents.forecasting.quant.binance_derivatives import attach_derivatives
            df = attach_derivatives(df, to_binance_symbol(asset))
        except Exception:
            pass
    return df
