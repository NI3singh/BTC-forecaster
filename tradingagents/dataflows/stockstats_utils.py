import logging
import os
import time
from typing import Annotated

import pandas as pd
import yfinance as yf
from stockstats import wrap
from yfinance.exceptions import YFRateLimitError

from .config import get_config
from .symbol_utils import NoMarketDataError, normalize_symbol
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)

# A vendor's latest OHLCV row this many calendar days before the requested date
# is treated as stale. Generous enough to span long holiday weekends, tight
# enough to catch the year-old frames yfinance occasionally returns (#1021).
MAX_OHLCV_STALE_DAYS = 10


def yf_retry(func, max_retries=3, base_delay=2.0):
    """Execute a yfinance call with exponential backoff on rate limits.

    yfinance raises YFRateLimitError on HTTP 429 responses but does not
    retry them internally. This wrapper adds retry logic specifically
    for rate limits. Other exceptions propagate immediately.
    """
    for attempt in range(max_retries + 1):
        try:
            return func()
        except YFRateLimitError:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Yahoo Finance rate limited, retrying in {delay:.0f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise


def _ensure_date_column(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize the date column to ``Date``.

    Some yfinance builds leave the index unnamed (so ``reset_index()`` yields
    ``index``) or use ``Datetime`` for intraday data. Rename the first
    date-like column so indicators don't silently drop when it isn't ``Date``.
    """
    if "Date" in data.columns:
        return data
    for candidate in ("index", "Datetime", "date"):
        if candidate in data.columns:
            return data.rename(columns={candidate: "Date"})
    return data


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize a stock DataFrame for stockstats: parse dates, drop invalid rows, fill price gaps."""
    data = _ensure_date_column(data)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date"])

    price_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data[price_cols] = data[price_cols].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["Close"])
    data[price_cols] = data[price_cols].ffill().bfill()

    return data


def _coerce_ohlcv_dates(data: pd.DataFrame) -> pd.Series:
    """Return parsed dates from an OHLCV frame, whether Date is a column or the index."""
    if "Date" in data.columns:
        return pd.to_datetime(data["Date"], errors="coerce").dropna()
    # yfinance keeps the dates in the index (a DatetimeIndex, sometimes unnamed).
    if isinstance(data.index, pd.DatetimeIndex):
        return pd.Series(pd.to_datetime(data.index, errors="coerce")).dropna()
    # Fallback: expose the index and look for any date-like column.
    df = data.reset_index()
    for col in ("Date", "Datetime", "date", "index"):
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce").dropna()
            if not parsed.empty:
                return parsed
    return pd.Series(dtype="datetime64[ns]")


def _assert_ohlcv_not_stale(
    data: pd.DataFrame,
    curr_date: str,
    symbol: str,
    canonical: str | None = None,
    *,
    max_stale_days: int = MAX_OHLCV_STALE_DAYS,
) -> None:
    """Reject OHLCV whose latest row is far older than curr_date.

    Raises NoMarketDataError (with a stale-specific detail) so the router treats
    it like any other "no usable data from this vendor" — try the next vendor,
    then emit one clear unavailable signal. Empty frames are left to the
    caller's existing no-data handling; this guards only the dangerous case of
    present-but-stale rows (a vendor returning a year-old frame that would
    otherwise feed wrong prices to the agent, #1021).
    """
    if data is None or data.empty:
        return
    requested = pd.to_datetime(curr_date, errors="coerce")
    if pd.isna(requested):
        return
    requested = requested.normalize()
    dates = _coerce_ohlcv_dates(data)
    if dates.empty:
        return
    latest = dates.max().normalize()
    stale_days = (requested - latest).days
    if stale_days > max_stale_days:
        raise NoMarketDataError(
            symbol,
            canonical,
            f"latest row is {latest.date()}, {stale_days} days before the "
            f"requested {requested.date()} (stale) — refusing to use it",
        )


# yfinance intraday interval -> a fixed pandas frequency for flooring timestamps.
# Used to decide when a same-day intraday cache has gone stale because a newer
# bar has closed since it was written. Daily bars don't churn intraday, so they
# are exempt from this check.
_INTERVAL_TO_FREQ = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min",
    "30m": "30min", "60m": "60min", "90m": "90min", "1h": "1h",
}

# yfinance caps how far back intraday history goes, per interval (in days). A
# request beyond the cap errors or returns partial data, so the fetch window is
# clamped to it. Sub-hourly bars (5m/15m/30m) max out at ~60 days; 1h at ~730.
_INTERVAL_MAX_DAYS = {
    "1m": 7, "2m": 59, "5m": 59, "15m": 59, "30m": 59, "90m": 59,
    "60m": 720, "1h": 720,
}


def floor_freq_for(interval: str) -> str:
    """Pandas floor frequency for a base bar interval (e.g. ``"5m"`` -> ``"5min"``).

    Aligns realized-price lookups and cache-freshness checks to the bar grid.
    Falls back to hourly for unknown / daily intervals.
    """
    return _INTERVAL_TO_FREQ.get(interval, "1h")


def max_intraday_days(interval: str) -> int:
    """Max look-back days yfinance allows for an intraday interval (else 720)."""
    return _INTERVAL_MAX_DAYS.get(interval, 720)


def _intraday_cache_is_stale(
    cached: pd.DataFrame, interval: str, curr_date_dt: pd.Timestamp
) -> bool:
    """True when a cached intraday frame predates the latest bar that should exist.

    Intraday caches are keyed by calendar date, so a frame written earlier in the
    day keeps serving old bars as new ones close — which silently freezes the
    forecast track record (``score`` can never see the realized price). Compare the
    cached frontier against the latest bar implied by the effective cutoff (the
    wall clock for a live ``curr_date``; an explicit backtest timestamp as-is) and
    report staleness so the caller refetches.
    """
    freq = _INTERVAL_TO_FREQ.get(interval)
    if freq is None or "Date" not in cached.columns:
        return True  # unknown intraday interval / shape — don't trust the cache
    dates = pd.to_datetime(cached["Date"], utc=True, errors="coerce").dt.tz_localize(None).dropna()
    if dates.empty:
        return True
    now = pd.Timestamp.utcnow().tz_localize(None)
    # Mirror the look-ahead cutoff used in load_ohlcv: a date-only (live) curr_date
    # tracks "now"; an explicit backtest timestamp is honored as-is.
    if curr_date_dt == curr_date_dt.normalize():
        cutoff = min(curr_date_dt + pd.Timedelta(hours=23, minutes=59, seconds=59), now)
    else:
        cutoff = curr_date_dt
    return dates.max() < cutoff.floor(freq)


def load_ohlcv(symbol: str, curr_date: str) -> pd.DataFrame:
    """Fetch OHLCV data with caching, filtered to prevent look-ahead bias.

    Downloads 5 years of data up to today and caches per symbol. On
    subsequent calls the cache is reused. Rows after curr_date are
    filtered out so backtests never see future prices.
    """
    # Resolve broker/forex symbols (XAUUSD+ -> GC=F) to Yahoo's convention,
    # then reject values that would escape the cache directory when
    # interpolated into the cache filename (e.g. ``../../tmp/x``).
    canonical = normalize_symbol(symbol)
    safe_symbol = safe_ticker_component(canonical)

    config = get_config()
    interval = config.get("data_interval", "1d")
    curr_date_dt = pd.to_datetime(curr_date)

    # Cache window: 5y for daily. yfinance caps intraday history per interval
    # (~60 days for 5m/15m/30m, ~730 for 1h), so clamp to the per-interval limit.
    # The interval is part of the cache filename so frames never collide.
    today_date = pd.Timestamp.today()
    if interval == "1d":
        start_date = today_date - pd.DateOffset(years=5)
    else:
        start_date = today_date - pd.Timedelta(days=max_intraday_days(interval))
    start_str = start_date.strftime("%Y-%m-%d")
    # yfinance ``end`` is EXCLUSIVE; request tomorrow so today's row is included
    # when curr_date is the current day (#986). Look-ahead is still prevented by
    # the curr_date filter below.
    end_str = (today_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(config["data_cache_dir"], exist_ok=True)
    data_file = os.path.join(
        config["data_cache_dir"],
        f"{safe_symbol}-YFin-{interval}-data-{start_str}-{end_str}.csv",
    )

    # A cached file may be empty if a prior fetch failed (unknown symbol,
    # transient rate limit). Treat an empty/columnless cache as a miss and
    # re-fetch rather than serving the poisoned file forever.
    data = None
    if os.path.exists(data_file):
        cached = pd.read_csv(data_file, on_bad_lines="skip", encoding="utf-8")
        if not cached.empty and "Close" in cached.columns:
            # An intraday cache written earlier the same day goes stale as new
            # bars close; refetch instead of serving a frozen frontier (which
            # would stop the forecast track record from ever resolving).
            if interval != "1d" and _intraday_cache_is_stale(cached, interval, curr_date_dt):
                data = None
            else:
                data = cached

    if data is None:
        downloaded = yf_retry(lambda: yf.download(
            canonical,
            start=start_str,
            end=end_str,
            interval=interval,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        downloaded = _ensure_date_column(downloaded.reset_index())
        # Only cache real data — never persist an empty frame.
        if downloaded.empty or "Close" not in downloaded.columns:
            raise NoMarketDataError(
                symbol, canonical, "Yahoo Finance returned no rows"
            )
        downloaded.to_csv(data_file, index=False, encoding="utf-8")
        data = downloaded

    data = _clean_dataframe(data)

    # Normalize timestamps to tz-naive UTC so intraday (tz-aware) and daily
    # (naive) frames both compare cleanly against the look-ahead cutoff below.
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="coerce").dt.tz_localize(None)
    data = data.dropna(subset=["Date"])

    # Filter to curr_date to prevent look-ahead bias in backtesting. Daily bars
    # cut at the date itself; intraday bars include every bar ON curr_date up to
    # "now" — a plain midnight cutoff would wrongly drop the current day's hourly
    # bars. An explicit time on curr_date (a backtest timestamp) is honored as-is.
    if interval == "1d" or curr_date_dt != curr_date_dt.normalize():
        cutoff = curr_date_dt
    else:
        day_end = curr_date_dt + pd.Timedelta(hours=23, minutes=59, seconds=59)
        cutoff = min(day_end, pd.Timestamp.utcnow().tz_localize(None))
    data = data[data["Date"] <= cutoff]

    # Reject a stale frame (latest row far older than curr_date) rather than
    # feeding year-old prices into indicators (#1021).
    _assert_ohlcv_not_stale(data, curr_date, symbol, canonical)

    return data


def filter_financials_by_date(data: pd.DataFrame, curr_date: str) -> pd.DataFrame:
    """Drop financial statement columns (fiscal period timestamps) after curr_date.

    yfinance financial statements use fiscal period end dates as columns.
    Columns after curr_date represent future data and are removed to
    prevent look-ahead bias.
    """
    if not curr_date or data.empty:
        return data
    cutoff = pd.Timestamp(curr_date)
    mask = pd.to_datetime(data.columns, errors="coerce") <= cutoff
    return data.loc[:, mask]


class StockstatsUtils:
    @staticmethod
    def get_stock_stats(
        symbol: Annotated[str, "ticker symbol for the company"],
        indicator: Annotated[
            str, "quantitative indicators based off of the stock data for the company"
        ],
        curr_date: Annotated[
            str, "curr date for retrieving stock price data, YYYY-mm-dd"
        ],
    ):
        interval = get_config().get("data_interval", "1d")
        data = load_ohlcv(symbol, curr_date)
        df = wrap(data)
        df[indicator]  # trigger stockstats to calculate the indicator

        if interval != "1d":
            # Intraday: load_ohlcv already trims to bars at/before the as_of
            # cutoff, so the most recent bar is the value we want.
            if df.empty:
                return "N/A: No intraday data at or before this time"
            return df.iloc[-1][indicator]

        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        curr_date_str = pd.to_datetime(curr_date).strftime("%Y-%m-%d")
        matching_rows = df[df["Date"].str.startswith(curr_date_str)]

        if not matching_rows.empty:
            indicator_value = matching_rows[indicator].values[0]
            return indicator_value
        else:
            return "N/A: Not a trading day (weekend or holiday)"
