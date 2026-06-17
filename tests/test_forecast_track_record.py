"""Tests for the forecast track-record / scoring loop (pure core, no network)."""

import pytest

from tradingagents.agents.schemas import Direction, Forecast, render_forecast
from tradingagents.forecasting.track_record import (
    ForecastTrackRecord,
    parse_forecast_markdown,
    realized_direction,
)


def _forecast_md():
    f = Forecast(
        direction_1h=Direction.UP, expected_price_1h=65950.0,
        range_low_1h=65700.0, range_high_1h=66150.0, confidence_1h=62,
        direction_4h=Direction.DOWN, expected_price_4h=65400.0,
        range_low_4h=64900.0, range_high_4h=65900.0, confidence_4h=48,
        reasons="MACD turned up; ATR ~250/hr.",
    )
    return render_forecast(f)


@pytest.mark.unit
class TestParseForecastMarkdown:
    def test_parses_both_horizons(self):
        preds = parse_forecast_markdown(_forecast_md())
        assert set(preds) == {"1h", "4h"}
        assert preds["1h"]["direction"] == "Up"
        assert preds["1h"]["expected_price"] == 65950.0
        assert preds["1h"]["range_low"] == 65700.0
        assert preds["1h"]["range_high"] == 66150.0
        assert preds["1h"]["confidence"] == 62
        assert preds["4h"]["direction"] == "Down"
        assert preds["4h"]["expected_price"] == 65400.0

    def test_handles_na_range(self):
        md = (
            "**Primary signal (next 1h): Flat**\n\n"
            "| Horizon | Direction | Approx. price | Expected range | Confidence |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Next 1h | Flat | $100.00 | n/a | Low (50%) |\n"
            "| Next 4h | Flat | $100.00 | n/a | Low (50%) |\n"
        )
        preds = parse_forecast_markdown(md)
        assert preds["1h"]["range_low"] is None
        assert preds["1h"]["range_high"] is None

    def test_empty_on_garbage(self):
        assert parse_forecast_markdown("no table here") == {}


@pytest.mark.unit
class TestRealizedDirection:
    def test_up_down_flat(self):
        assert realized_direction(100.0, 101.0) == "Up"
        assert realized_direction(100.0, 99.0) == "Down"
        assert realized_direction(100.0, 100.05) == "Flat"  # inside deadband


@pytest.mark.unit
class TestForecastTrackRecord:
    def test_log_is_idempotent(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(_forecast_md())
        assert tr.log("BTC-USD", "2026-06-17T05:00:00", preds, current_price=65800.0) is True
        assert tr.log("BTC-USD", "2026-06-17T05:00:00", preds, current_price=65800.0) is False
        assert tr.summary()["total_forecasts"] == 1

    def test_resolve_scores_elapsed_horizons(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(_forecast_md())
        tr.log("BTC-USD", "2026-06-17T05:00:00", preds, current_price=65800.0)

        # Realized: +1h price went up into the predicted range; +4h not elapsed yet.
        def price_at(asset, target_iso):
            return 66000.0 if target_iso.startswith("2026-06-17T06:00") else None

        assert tr.resolve(price_at) == 1
        s = tr.summary()["by_horizon"]
        assert s["1h"]["resolved"] == 1
        assert s["1h"]["directional_accuracy"] == 1.0  # predicted Up, realized Up
        assert s["1h"]["range_coverage"] == 1.0  # 66000 within 65700-66150
        assert s["4h"]["resolved"] == 0  # not elapsed

    def test_resolve_marks_wrong_direction(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(_forecast_md())
        tr.log("BTC-USD", "2026-06-17T05:00:00", preds, current_price=65800.0)

        # Predicted 1h Up, but realized price fell -> wrong; outside range.
        def price_at(asset, target_iso):
            return 64000.0 if "T06:00" in target_iso else None

        tr.resolve(price_at)
        s = tr.summary()["by_horizon"]
        assert s["1h"]["directional_accuracy"] == 0.0
        assert s["1h"]["range_coverage"] == 0.0

    def test_resolve_is_repeatable_without_double_counting(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(_forecast_md())
        tr.log("BTC-USD", "2026-06-17T05:00:00", preds, current_price=65800.0)

        def price_at(asset, target_iso):
            return 66000.0 if "T06:00" in target_iso else None

        assert tr.resolve(price_at) == 1
        assert tr.resolve(price_at) == 0  # already scored, not re-counted

    def test_summary_markdown_renders(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        md = tr.summary_markdown()
        assert "Forecast track record" in md
        assert "Directional acc." in md
