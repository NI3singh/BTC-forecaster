"""Tests for the deterministic volatility-scaled range override (pure, no network)."""

import pytest

from tests._forecast_helpers import make_forecast
from tradingagents.agents.schemas import FORECAST_HORIZONS, Direction
from tradingagents.forecasting.ranges import (
    apply_vol_scaled_ranges,
    render_anchor_block,
    vol_scaled_band,
)


@pytest.mark.unit
class TestVolScaledBand:
    def test_half_width_at_reference_bar(self):
        # 5m horizon = 1 base bar: hw = k*sigma*sqrt(1) = 1.5*0.01 = 0.015.
        low, high = vol_scaled_band(100.0, 0.01, 5, k=1.5)
        assert low == pytest.approx(98.5)
        assert high == pytest.approx(101.5)

    def test_half_width_scales_with_sqrt_time(self):
        # 20m = 4 bars -> sqrt(4) = 2x the 5m half-width.
        low, high = vol_scaled_band(100.0, 0.01, 20, k=1.5)
        assert low == pytest.approx(97.0)
        assert high == pytest.approx(103.0)


@pytest.mark.unit
class TestApplyVolScaledRanges:
    def test_widens_monotonically_and_brackets_expected(self):
        f = apply_vol_scaled_ranges(make_forecast(), sigma_bar=0.001)
        widths = []
        for label, _ in FORECAST_HORIZONS:
            low = getattr(f, f"range_low_{label}")
            high = getattr(f, f"range_high_{label}")
            expected = getattr(f, f"expected_price_{label}")
            assert low <= expected <= high
            widths.append(high - low)
        assert widths == sorted(widths)       # non-decreasing across horizons
        assert widths[-1] > widths[0]         # 4h band strictly wider than 5m

    def test_preserves_direction_expected_and_confidence(self):
        original = make_forecast(direction_4h=Direction.DOWN, confidence_1h=73)
        f = apply_vol_scaled_ranges(original, sigma_bar=0.002)
        assert f.direction_4h == Direction.DOWN
        assert f.expected_price_1h == 65000.0
        assert f.confidence_1h == 73

    @pytest.mark.parametrize("bad", [0.0, -0.01, float("nan"), None])
    def test_noop_on_bad_sigma(self, bad):
        f = apply_vol_scaled_ranges(make_forecast(), sigma_bar=bad)
        assert f.range_low_5m == 64500.0   # untouched fixture default
        assert f.range_high_5m == 65500.0


@pytest.mark.unit
class TestRenderAnchorBlock:
    def test_full_block(self):
        block = render_anchor_block(
            "BTC-USD",
            {"spot": 65000.0, "sigma_bar": 0.001, "recent_low": 64000.0, "recent_high": 66000.0},
        )
        assert "$65,000.00" in block
        assert "Spot price" in block
        assert "Realized volatility" in block
        assert "Recent range" in block

    def test_minimal_block_omits_missing_fields(self):
        block = render_anchor_block("BTC-USD", {"spot": 65000.0})
        assert "$65,000.00" in block
        assert "Realized volatility" not in block
        assert "Recent range" not in block
