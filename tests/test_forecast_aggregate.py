"""Tests for Portfolio-Manager self-consistency aggregation (pure, no network)."""

import pytest

from tests._forecast_helpers import make_forecast
from tradingagents.agents.schemas import Direction, aggregate_forecasts, render_forecast
from tradingagents.agents.utils.structured import invoke_structured_or_freetext


@pytest.mark.unit
class TestAggregateForecasts:
    def test_single_sample_returned_unchanged(self):
        f = make_forecast()
        assert aggregate_forecasts([f]) is f

    def test_majority_direction_and_confidence_from_agreement(self):
        fs = [
            make_forecast(direction_1h=Direction.UP),
            make_forecast(direction_1h=Direction.UP),
            make_forecast(direction_1h=Direction.DOWN),
        ]
        agg = aggregate_forecasts(fs)
        assert agg.direction_1h == Direction.UP
        assert agg.confidence_1h == 67  # round(100 * 2/3) — agreement fraction

    def test_median_price_and_ranges(self):
        fs = [
            make_forecast(expected_price_1h=100, range_low_1h=90, range_high_1h=110),
            make_forecast(expected_price_1h=110, range_low_1h=95, range_high_1h=120),
            make_forecast(expected_price_1h=120, range_low_1h=100, range_high_1h=130),
        ]
        agg = aggregate_forecasts(fs)
        assert agg.expected_price_1h == 110
        assert agg.range_low_1h == 95
        assert agg.range_high_1h == 120

    def test_tie_broken_by_summed_confidence(self):
        fs = [
            make_forecast(direction_1h=Direction.UP, confidence_1h=55),
            make_forecast(direction_1h=Direction.DOWN, confidence_1h=90),
        ]
        agg = aggregate_forecasts(fs)
        assert agg.direction_1h == Direction.DOWN  # 1-1 tie -> higher total confidence
        assert agg.confidence_1h == 50

    def test_prose_taken_from_consensus_matching_sample(self):
        fs = [
            make_forecast(reasons="A"),                              # all Up
            make_forecast(reasons="B"),                              # all Up
            make_forecast(direction_1h=Direction.DOWN, reasons="C"),  # differs at 1h
        ]
        agg = aggregate_forecasts(fs)
        assert agg.reasons == "A"  # first sample matching the all-Up consensus


@pytest.mark.unit
class TestInvokeWithSampling:
    def _fake(self, forecasts):
        it = iter(forecasts)

        class Fake:
            def invoke(self, prompt):
                return next(it)

        return Fake()

    def test_samples_and_aggregates(self):
        fake = self._fake([
            make_forecast(direction_1h=Direction.UP),
            make_forecast(direction_1h=Direction.UP),
            make_forecast(direction_1h=Direction.DOWN),
        ])
        md = invoke_structured_or_freetext(
            fake, None, "p", render_forecast, "PM",
            samples=3, aggregate=aggregate_forecasts,
        )
        assert "Primary signal (next 1h): Up" in md  # 2-of-3 consensus

    def test_single_sample_path_unchanged(self):
        fake = self._fake([make_forecast(direction_1h=Direction.DOWN)])
        md = invoke_structured_or_freetext(fake, None, "p", render_forecast, "PM")
        assert "Primary signal (next 1h): Down" in md
