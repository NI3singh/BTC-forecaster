"""Tests for the forecast track-record / scoring loop (pure core, no network)."""

import pytest

from tests._forecast_helpers import make_forecast
from tradingagents.agents.schemas import Direction, render_forecast
from tradingagents.forecasting.track_record import (
    ForecastTrackRecord,
    deadband_for,
    forecast_feedback_block,
    parse_forecast_markdown,
    realized_direction,
    wilson_interval,
)


def _forecast_md():
    f = make_forecast(
        direction_1h=Direction.UP, expected_price_1h=65950.0,
        range_low_1h=65700.0, range_high_1h=66150.0, confidence_1h=62,
        direction_4h=Direction.DOWN, expected_price_4h=65400.0,
        range_low_4h=64900.0, range_high_4h=65900.0, confidence_4h=48,
    )
    return render_forecast(f)


@pytest.mark.unit
class TestParseForecastMarkdown:
    def test_parses_all_horizons(self):
        preds = parse_forecast_markdown(_forecast_md())
        assert set(preds) == {"5m", "15m", "30m", "1h", "2h", "4h"}
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
        # 66100 is +0.46% over the 65800 entry — clear of the ~0.35% 1h deadband.
        def price_at(asset, target_iso):
            return 66100.0 if target_iso.startswith("2026-06-17T06:00") else None

        assert tr.resolve(price_at) == 1
        s = tr.summary()["by_horizon"]
        assert s["1h"]["resolved"] == 1
        assert s["1h"]["directional_accuracy"] == 1.0  # predicted Up, realized Up
        assert s["1h"]["range_coverage"] == 1.0  # 66100 within 65700-66150
        assert s["4h"]["resolved"] == 0  # not elapsed

    def test_resolve_scores_minute_horizon(self, tmp_path):
        # Sub-hourly horizons must resolve via timedelta(minutes=...): the 5m
        # target is as_of + 5 minutes (05:05), not an hour-floored 05:00.
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(_forecast_md())
        tr.log("BTC-USD", "2026-06-17T05:00:00", preds, current_price=65000.0)

        def price_at(asset, target_iso):
            return 65200.0 if target_iso.startswith("2026-06-17T05:05") else None

        assert tr.resolve(price_at) == 1
        s = tr.summary()["by_horizon"]
        assert s["5m"]["resolved"] == 1
        assert s["5m"]["directional_accuracy"] == 1.0  # default Up, realized up
        assert s["15m"]["resolved"] == 0  # +15m not elapsed

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


@pytest.mark.unit
class TestDeadbandFor:
    def test_reference_horizon_is_unscaled(self):
        assert deadband_for(5) == pytest.approx(0.001)

    def test_scales_with_sqrt_time(self):
        # 20 minutes is 4x the 5m reference, so the band is sqrt(4) = 2x.
        assert deadband_for(20, base=0.002, ref_minutes=5) == pytest.approx(0.004)

    def test_longer_horizons_get_wider_bands(self):
        bands = [deadband_for(m) for m in (5, 15, 30, 60, 120, 240)]
        assert bands == sorted(bands)  # monotonically non-decreasing
        assert bands[-1] > bands[0]

    def test_non_positive_minutes_falls_back_to_base(self):
        assert deadband_for(0, base=0.001) == 0.001


@pytest.mark.unit
class TestWilsonInterval:
    def test_known_value(self):
        lo, hi = wilson_interval(6, 10)
        assert lo == pytest.approx(0.3127, abs=0.01)
        assert hi == pytest.approx(0.8319, abs=0.01)
        assert lo < 0.6 < hi

    def test_empty_sample_is_fully_uncertain(self):
        assert wilson_interval(0, 0) == (0.0, 1.0)

    def test_perfect_sample_caps_at_one(self):
        lo, hi = wilson_interval(10, 10)
        assert hi == 1.0
        assert lo < 1.0


@pytest.mark.unit
class TestBaselinesAndCalibration:
    """Four 1h forecasts: 2 right (Up) + 2 wrong (Down), all at 60% confidence."""

    def _logged_and_resolved(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(
            render_forecast(make_forecast(direction_1h=Direction.UP, confidence_1h=60))
        )
        for day in ("14", "15", "16", "17"):
            tr.log("BTC-USD", f"2026-06-{day}T05:00:00", preds, current_price=65000.0)

        # 1h target is dayT06:00. First two days rise (Up, correct), last two fall.
        realized = {"2026-06-14": 66000.0, "2026-06-15": 66000.0,
                    "2026-06-16": 64000.0, "2026-06-17": 64000.0}

        def price_at(asset, target_iso):
            if "T06:00" in target_iso:
                return realized.get(target_iso[:10])
            return None

        assert tr.resolve(price_at) == 4  # one 1h horizon per day
        return tr.summary()["by_horizon"]["1h"]

    def test_directional_accuracy_and_ci(self, tmp_path):
        st = self._logged_and_resolved(tmp_path)
        assert st["directional_accuracy"] == 0.5
        lo, hi = st["directional_accuracy_ci"]
        assert lo < 0.5 < hi
        assert st["sufficient_n"] is False  # only 4 resolved

    def test_brier_and_calibration_gap(self, tmp_path):
        st = self._logged_and_resolved(tmp_path)
        # p=0.6, outcomes [1,1,0,0] -> mean(0.16,0.16,0.36,0.36) = 0.26
        assert st["brier"] == pytest.approx(0.26)
        assert st["mean_confidence"] == pytest.approx(0.6)
        assert st["calibration_gap"] == pytest.approx(0.1)  # overconfident by 10pp

    def test_reliability_bin(self, tmp_path):
        st = self._logged_and_resolved(tmp_path)
        assert len(st["reliability"]) == 1
        band = st["reliability"][0]
        assert band["band"] == "60-70%"
        assert band["n"] == 4
        assert band["empirical_accuracy"] == pytest.approx(0.5)

    def test_baselines_and_skill(self, tmp_path):
        st = self._logged_and_resolved(tmp_path)
        assert st["baselines"]["flat"] == 0.0        # no realized move was Flat
        assert st["baselines"]["momentum"] == 0.0    # flat entries predict Flat
        assert st["baselines"]["majority"] == 0.5    # Up is most common (2/4)
        assert st["best_baseline"] == "majority"
        assert st["skill_vs_best_baseline"] == pytest.approx(0.0)


@pytest.mark.unit
class TestForecastFeedbackBlock:
    def _log_and_resolve(self, tmp_path, n):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        preds = parse_forecast_markdown(render_forecast(make_forecast(confidence_1h=70)))
        realized = {}
        for i in range(n):
            day = f"2026-06-{i + 1:02d}"
            tr.log("BTC-USD", f"{day}T05:00:00", preds, current_price=65000.0)
            realized[day] = 66000.0 if i % 2 == 0 else 64000.0  # alternate Up/Down

        def price_at(asset, target_iso):
            return realized.get(target_iso[:10]) if "T06:00" in target_iso else None

        tr.resolve(price_at)
        return tmp_path / "tr.jsonl"

    def test_empty_when_no_log(self, tmp_path):
        assert forecast_feedback_block({"forecast_log_path": str(tmp_path / "nope.jsonl")}) == ""

    def test_empty_when_too_few_resolved(self, tmp_path):
        path = self._log_and_resolve(tmp_path, 3)  # below the 10-resolved gate
        assert forecast_feedback_block({"forecast_log_path": str(path)}) == ""

    def test_returns_per_horizon_feedback_when_enough(self, tmp_path):
        path = self._log_and_resolve(tmp_path, 12)
        block = forecast_feedback_block({"forecast_log_path": str(path)})
        assert "1h:" in block
        assert "directional" in block
        assert "n=12" in block
        assert "calibrate" in block.lower()
        # The 5m horizon never resolved (price_at only answered the 1h target).
        assert "5m:" not in block


@pytest.mark.unit
class TestDirectionalCommitment:
    """Five 1h calls exercising every commit/move combination at entry 65000.

    The 1h deadband is ~0.35%, so >65225 is Up, <64775 is Down, else Flat.

      day  predicted  realized  -> committed?  moved?  side-correct?
      14   Up         66000(Up)    yes          yes     yes
      15   Up         64000(Down)  yes          yes     no
      16   Down       64000(Down)  yes          yes     yes
      17   Up         65000(Flat)  yes          no      (excluded)
      18   Flat       65000(Flat)  no           -       -
    """

    def _resolve(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        plan = {
            "14": Direction.UP, "15": Direction.UP, "16": Direction.DOWN,
            "17": Direction.UP, "18": Direction.FLAT,
        }
        for day, direction in plan.items():
            preds = parse_forecast_markdown(
                render_forecast(make_forecast(direction_1h=direction))
            )
            tr.log("BTC-USD", f"2026-06-{day}T05:00:00", preds, current_price=65000.0)

        realized = {"2026-06-14": 66000.0, "2026-06-15": 64000.0, "2026-06-16": 64000.0,
                    "2026-06-17": 65000.0, "2026-06-18": 65000.0}

        def price_at(asset, target_iso):
            if "T06:00" in target_iso:
                return realized.get(target_iso[:10])
            return None

        tr.resolve(price_at)
        return tr.summary()["by_horizon"]["1h"]

    def test_committed_rate_and_accuracy(self, tmp_path):
        st = self._resolve(tmp_path)
        assert st["committed_n"] == 4            # all but the Flat prediction
        assert st["committed_rate"] == pytest.approx(0.8)   # 4 of 5 calls
        assert st["committed_accuracy"] == pytest.approx(0.5)  # 2 of 4 landed

    def test_right_side_accuracy_excludes_flat_outcomes(self, tmp_path):
        st = self._resolve(tmp_path)
        # Only the 3 committed calls where the market actually moved count; day 17
        # (committed Up but realized Flat) is excluded. 2 of those 3 were correct.
        assert st["side_n"] == 3
        assert st["side_accuracy"] == pytest.approx(2 / 3)
        lo, hi = st["side_accuracy_ci"]
        assert lo < 2 / 3 < hi

    def test_overall_accuracy_is_higher_due_to_easy_flat_call(self, tmp_path):
        st = self._resolve(tmp_path)
        # Overall counts the easy Flat=Flat hit (day 18): 3 of 5 = 60%, above the
        # 2/3 right-side number's lower bound — the point of stratifying.
        assert st["directional_accuracy"] == pytest.approx(0.6)

    def test_markdown_has_commitment_section(self, tmp_path):
        tr = ForecastTrackRecord(tmp_path / "tr.jsonl")
        assert "Right-side acc." in tr.summary_markdown()
