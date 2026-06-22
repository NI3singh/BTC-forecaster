"""Stale OHLCV guard (#1021): a vendor returning a year-old partial frame must
be rejected, not fed into the report as if it were current.

The guard raises NoMarketDataError with a stale-specific detail, so the router's
existing try-next-vendor + single-sentinel handling applies and the sentinel
surfaces the reason.
"""
import copy
import unittest
from unittest import mock

import pandas as pd
import pytest

import tradingagents.dataflows.config as config_module
import tradingagents.dataflows.y_finance as y_finance
import tradingagents.default_config as default_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.stockstats_utils import (
    _assert_ohlcv_not_stale,
    _intraday_cache_is_stale,
)
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _frame(date):
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(date)],
            "Open": [330.0],
            "High": [332.0],
            "Low": [328.0],
            "Close": [330.58],
            "Volume": [1_000_000],
        }
    )


@pytest.mark.unit
class StaleGuardUnitTests(unittest.TestCase):
    def test_recent_prior_trading_day_is_accepted(self):
        # 1 day before curr_date — well within the freshness window.
        _assert_ohlcv_not_stale(_frame("2026-06-10"), "2026-06-11", "CB")

    def test_year_old_row_is_rejected_with_detail(self):
        with self.assertRaises(NoMarketDataError) as ctx:
            _assert_ohlcv_not_stale(_frame("2025-06-11"), "2026-06-11", "CB", "CB")
        msg = str(ctx.exception)
        self.assertIn("2025-06-11", msg)
        self.assertIn("2026-06-11", msg)
        self.assertIn("stale", msg)

    def test_empty_frame_is_left_to_caller(self):
        # Empty is a no-data condition handled elsewhere, not a staleness one.
        _assert_ohlcv_not_stale(
            pd.DataFrame(columns=["Date", "Close"]), "2026-06-11", "X"
        )

    def test_long_holiday_gap_within_threshold_is_accepted(self):
        _assert_ohlcv_not_stale(_frame("2026-06-02"), "2026-06-11", "X")  # 9 days


@pytest.mark.unit
class StaleGuardPropagationTests(unittest.TestCase):
    def test_get_yfin_data_online_raises_on_stale_frame(self):
        stale = pd.DataFrame(
            {
                "Open": [280.0], "High": [286.0], "Low": [278.0],
                "Close": [284.45], "Volume": [1_000_000],
            },
            index=pd.DatetimeIndex([pd.Timestamp("2025-06-11")], name="Date"),
        )

        class DummyTicker:
            def __init__(self, symbol):
                pass

            def history(self, start, end, interval=None):
                return stale

        with mock.patch.object(y_finance.yf, "Ticker", DummyTicker), \
                self.assertRaises(NoMarketDataError):
            y_finance.get_YFin_data_online("CB", "2026-06-01", "2026-06-11")


@pytest.mark.unit
class StaleGuardRoutingTests(unittest.TestCase):
    def setUp(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def tearDown(self):
        config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)

    def test_router_sentinel_surfaces_stale_reason(self):
        set_config({"data_vendors": {"core_stock_apis": "yfinance"}})

        def _stale(symbol, *a, **k):
            raise NoMarketDataError(
                symbol, symbol, "latest row is 2025-06-11, 365 days before ... (stale)"
            )

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"yfinance": _stale}},
            clear=False,
        ):
            out = interface.route_to_vendor(
                "get_stock_data", "CB", "2026-06-01", "2026-06-11"
            )
        self.assertIn("NO_DATA_AVAILABLE", out)
        self.assertIn("stale", out)  # the typed detail is surfaced to the agent


def _intraday_frame(*timestamps):
    n = len(timestamps)
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(t) for t in timestamps],
            "Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n,
            "Close": [1.0] * n, "Volume": [1] * n,
        }
    )


@pytest.mark.unit
class IntradayCacheFreshnessTests(unittest.TestCase):
    """An intraday cache is keyed only by calendar date, so a frame written
    earlier the same day must be refetched once newer bars have closed —
    otherwise the forecast track record can never see the realized price.
    """

    def test_live_same_day_stale_cache_is_refetched(self):
        # Live path: a date-only curr_date tracks the wall clock. A cache whose
        # latest bar is hours behind the current bar is stale.
        now = pd.Timestamp.utcnow().tz_localize(None)
        cached = _intraday_frame(now.floor("h") - pd.Timedelta(hours=3))
        self.assertTrue(_intraday_cache_is_stale(cached, "1h", now.normalize()))

    def test_cache_behind_cutoff_is_stale(self):
        # An explicit timestamp (backtest) is honored as-is; a cache ending before
        # the floored cutoff is stale.
        cached = _intraday_frame("2026-06-17 06:00:00")
        self.assertTrue(
            _intraday_cache_is_stale(cached, "1h", pd.Timestamp("2026-06-17 09:54:00"))
        )

    def test_cache_at_cutoff_is_fresh(self):
        cached = _intraday_frame("2026-06-17 09:00:00")  # == floor(09:54)
        self.assertFalse(
            _intraday_cache_is_stale(cached, "1h", pd.Timestamp("2026-06-17 09:54:00"))
        )

    def test_unknown_interval_is_treated_as_stale(self):
        # 4h isn't a yfinance bar size we floor on — don't trust a same-day cache.
        cached = _intraday_frame("2026-06-17 09:00:00")
        self.assertTrue(
            _intraday_cache_is_stale(cached, "4h", pd.Timestamp("2026-06-17 09:54:00"))
        )

    def test_missing_or_empty_dates_treated_as_stale(self):
        cutoff = pd.Timestamp("2026-06-17 09:54:00")
        self.assertTrue(_intraday_cache_is_stale(pd.DataFrame({"Close": [1.0]}), "1h", cutoff))
        self.assertTrue(
            _intraday_cache_is_stale(pd.DataFrame({"Date": [], "Close": []}), "1h", cutoff)
        )


@pytest.mark.unit
class IntervalHelperTests(unittest.TestCase):
    """Per-interval helpers underpinning the sub-hourly (5m/15m/30m) data path."""

    def test_floor_freq_for(self):
        from tradingagents.dataflows.stockstats_utils import floor_freq_for
        self.assertEqual(floor_freq_for("5m"), "5min")
        self.assertEqual(floor_freq_for("30m"), "30min")
        self.assertEqual(floor_freq_for("1h"), "1h")
        self.assertEqual(floor_freq_for("1d"), "1h")  # fallback for non-intraday

    def test_max_intraday_days(self):
        from tradingagents.dataflows.stockstats_utils import max_intraday_days
        self.assertEqual(max_intraday_days("5m"), 59)
        self.assertEqual(max_intraday_days("15m"), 59)
        self.assertEqual(max_intraday_days("1h"), 720)
        self.assertEqual(max_intraday_days("unknown"), 720)  # fallback


if __name__ == "__main__":
    unittest.main()
