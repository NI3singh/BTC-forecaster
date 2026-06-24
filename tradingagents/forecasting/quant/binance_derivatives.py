"""Binance USD-M Futures derivatives + order-flow data for the quant model.

Funding rate, open interest, and long/short positioning — the exogenous signals
spot OHLCV cannot see, and the highest-ROI inputs for crypto intraday direction.
Same raw-REST + growing-CSV-cache style as ``binance_data.py``; no API key needed.

The ``/futures/data/`` endpoints (OI, long/short) only retain ~30 days, so their
caches GROW over time and older bars simply lack these columns — the gradient
boosting model handles the resulting NaN natively. Funding has long history.

LEAKAGE: every reading is aligned to a bar via ``reindex(..., method="ffill")``,
i.e. the last value at or before the bar timestamp, so a feature for bar t only
uses data known by t.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

FAPI = "https://fapi.binance.com"
_DAY_MS = 86_400_000


def _cache_path(symbol: str, name: str) -> Path:
    from tradingagents.dataflows.config import get_config
    d = Path(get_config().get("data_cache_dir", ".")) / "quant"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{symbol}_{name}.csv"


def _get(path: str, params: dict) -> list:
    resp = requests.get(FAPI + path, params=params, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _paginate(path: str, params: dict, days: int, page_limit: int, ts_field: str) -> list:
    """Walk BACKWARD by ``endTime`` (like ``fetch_klines``) until ``days`` is covered.

    The ``/futures/data/`` endpoints return the most-recent page when given only a
    limit, so we page back from now via ``endTime`` rather than forward from a
    startTime (which they don't honor at the retention edge).
    """
    target = days * _DAY_MS
    end_time: int | None = None
    newest: int | None = None
    rows: list = []
    for _ in range(400):  # safety cap on pages
        p = dict(params, limit=page_limit)
        if end_time is not None:
            p["endTime"] = end_time
        batch = _get(path, p)
        if not batch:
            break
        rows = batch + rows
        if newest is None:
            newest = int(batch[-1][ts_field])
        oldest = int(batch[0][ts_field])
        if len(batch) < page_limit or newest - oldest >= target:
            break
        end_time = oldest - 1
        time.sleep(0.2)
    return rows


def _to_frame(rows: list, ts_field: str, value_map: dict[str, str]) -> pd.DataFrame:
    """Parse rows into a UTC-timestamp-indexed frame with renamed numeric columns."""
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df[ts_field].astype("int64"), unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    out = pd.DataFrame(index=df.index)
    for src, dst in value_map.items():
        out[dst] = pd.to_numeric(df[src], errors="coerce")
    return out[~out.index.duplicated(keep="last")]


def _load_cache(symbol: str, name: str) -> pd.DataFrame | None:
    path = _cache_path(symbol, name)
    if path.exists():
        df = pd.read_csv(path, index_col="timestamp")
        # Coerce explicitly with ISO8601: funding's mixed sub-second precision
        # (some rows ".002000", some none) defeats read_csv's parse_dates and a
        # plain to_datetime (which locks the format from row 0), leaving an object
        # index that breaks reindex.
        df.index = pd.to_datetime(df.index, utc=True, format="ISO8601")
        return df
    return None


def _merge_cache(symbol: str, name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Append new rows to the growing cache (dedup + sort on the timestamp index)."""
    if df.empty:
        return df
    path = _cache_path(symbol, name)
    old = _load_cache(symbol, name)
    if old is not None:
        df = pd.concat([old, df])
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_csv(path)
    return df


def fetch_funding(symbol: str, days: int) -> pd.DataFrame:
    rows = _paginate("/fapi/v1/fundingRate", {"symbol": symbol}, days, 1000, "fundingTime")
    return _to_frame(rows, "fundingTime", {"fundingRate": "funding_rate"})


def fetch_open_interest(symbol: str, period: str, days: int) -> pd.DataFrame:
    rows = _paginate("/futures/data/openInterestHist",
                     {"symbol": symbol, "period": period}, days, 500, "timestamp")
    return _to_frame(rows, "timestamp",
                     {"sumOpenInterest": "sum_oi", "sumOpenInterestValue": "sum_oi_value"})


def fetch_long_short(symbol: str, path: str, col: str, period: str, days: int) -> pd.DataFrame:
    rows = _paginate(path, {"symbol": symbol, "period": period}, days, 500, "timestamp")
    return _to_frame(rows, "timestamp", {"longShortRatio": col})


def fetch_taker_long_short(symbol: str, period: str, days: int) -> pd.DataFrame:
    rows = _paginate("/futures/data/takerlongshortRatio",
                     {"symbol": symbol, "period": period}, days, 500, "timestamp")
    return _to_frame(rows, "timestamp", {"buySellRatio": "taker_ls_ratio"})


def attach_derivatives(df: pd.DataFrame, symbol: str, period: str = "5m") -> pd.DataFrame:
    """Best-effort: fetch funding/OI/long-short, align onto ``df``'s index (ffill,
    leakage-safe), and join. Each source is independent — any failure just leaves
    that source's columns absent, so the model falls back to its current inputs.
    """
    from tradingagents.dataflows.config import get_config
    oi_days = int(get_config().get("quant_oi_days", 30))
    # Funding has full history; cover the whole kline window so it isn't all-NaN.
    span_days = max(oi_days, (df.index[-1] - df.index[0]).days + 2) if len(df) else oi_days

    sources = [
        ("funding", lambda: fetch_funding(symbol, span_days)),
        ("oi_5m", lambda: fetch_open_interest(symbol, period, oi_days)),
        ("gls_5m", lambda: fetch_long_short(
            symbol, "/futures/data/globalLongShortAccountRatio", "gls_ratio", period, oi_days)),
        ("tps_5m", lambda: fetch_long_short(
            symbol, "/futures/data/topLongShortPositionRatio", "tps_ratio", period, oi_days)),
        ("taker_5m", lambda: fetch_taker_long_short(symbol, period, oi_days)),
    ]
    out = df
    for name, fn in sources:
        try:
            try:
                fetched = fn()
            except Exception:
                fetched = None
            # A fresh fetch grows the cache; on failure fall back to the cache so a
            # transient rate-limit doesn't silently drop an already-downloaded source.
            if fetched is not None and not fetched.empty:
                data = _merge_cache(symbol, name, fetched)
            else:
                data = _load_cache(symbol, name)
            if data is None or data.empty:
                continue
            out = out.join(data.reindex(out.index, method="ffill"))
        except Exception:
            continue  # best-effort; this source's columns stay absent
    return out
