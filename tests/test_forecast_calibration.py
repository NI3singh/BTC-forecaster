"""Tests for post-hoc confidence calibration (pure isotonic core, no network)."""

import json

import pytest

from tests._forecast_helpers import make_forecast
from tradingagents.agents.schemas import render_forecast
from tradingagents.forecasting.calibration import (
    CalibrationMap,
    _pav,
    calibrate_markdown,
    fit_and_save,
    load_map,
)


def _records(n, conf=80, horizon="1h"):
    """n minimal resolved records at one horizon; even indices are correct."""
    return [
        {
            "asset": "BTC-USD",
            "as_of": f"2026-06-01T00:{i:02d}:00",
            "current_price": 65000.0,
            "horizons": {horizon: {"confidence": conf, "direction_correct": (i % 2 == 0)}},
        }
        for i in range(n)
    ]


@pytest.mark.unit
class TestPav:
    def test_enforces_monotonic_nondecreasing(self):
        fitted = _pav([1.0, 0.0, 1.0, 0.0])
        assert fitted == sorted(fitted)        # non-decreasing
        assert fitted[0] == fitted[-1]         # all pooled to the mean (0.5)
        assert fitted[0] == pytest.approx(0.5)

    def test_already_sorted_is_unchanged(self):
        assert _pav([0.0, 0.0, 1.0, 1.0]) == [0.0, 0.0, 1.0, 1.0]


@pytest.mark.unit
class TestCalibrationMap:
    def test_corrects_overconfidence(self):
        # 60 calls all at 80% confidence but only 50% right -> 80 maps to ~50.
        cmap = CalibrationMap.fit_from_records(_records(60, conf=80), min_n=10)
        assert not cmap.is_empty()
        assert cmap.calibrate(80, "1h") == pytest.approx(50, abs=3)

    def test_identity_below_min_samples(self):
        cmap = CalibrationMap.fit_from_records(_records(8, conf=80), min_n=10)
        assert cmap.is_empty()
        assert cmap.calibrate(80, "1h") == 80      # untouched
        assert cmap.calibrate(90, "4h") == 90      # unmapped horizon

    def test_calibrate_is_monotonic_nondecreasing(self):
        # 50%-confidence calls right 20% of the time; 90%-confidence right 80%.
        lo = [
            {"asset": "X", "as_of": f"a{i}", "current_price": 1.0,
             "horizons": {"1h": {"confidence": 50, "direction_correct": i < 6}}}
            for i in range(30)
        ]
        hi = [
            {"asset": "X", "as_of": f"b{i}", "current_price": 1.0,
             "horizons": {"1h": {"confidence": 90, "direction_correct": i < 24}}}
            for i in range(30)
        ]
        cmap = CalibrationMap.fit_from_records(lo + hi, min_n=10)
        vals = [cmap.calibrate(c, "1h") for c in range(0, 101, 10)]
        assert vals == sorted(vals)                                   # non-decreasing
        assert cmap.calibrate(90, "1h") > cmap.calibrate(50, "1h")    # 80% vs 20%

    def test_dict_and_disk_round_trip(self, tmp_path):
        cmap = CalibrationMap.fit_from_records(_records(60), min_n=10)
        assert CalibrationMap.from_dict(cmap.to_dict()).to_dict() == cmap.to_dict()
        p = tmp_path / "cal.json"
        cmap.save(p)
        assert CalibrationMap.load(p).calibrate(80, "1h") == cmap.calibrate(80, "1h")

    def test_load_missing_file_is_empty(self, tmp_path):
        assert CalibrationMap.load(tmp_path / "nope.json").is_empty()


@pytest.mark.unit
class TestCalibrateMarkdown:
    def test_rewrites_confidence_for_mapped_horizon_only(self):
        md = render_forecast(make_forecast(confidence_1h=62))
        # A map that sends every 1h confidence to 0.5 (Low/50%).
        cmap = CalibrationMap({"1h": [[0.0, 0.5], [1.0, 0.5]]})
        out = calibrate_markdown(md, cmap)
        assert "(50%)" in out      # 1h recalibrated 62 -> 50
        assert "(62%)" not in out  # original 1h value replaced
        assert "(55%)" in out      # other horizons (unmapped) untouched

    def test_empty_map_is_noop(self):
        md = render_forecast(make_forecast(confidence_1h=62))
        assert calibrate_markdown(md, CalibrationMap()) == md


@pytest.mark.unit
class TestFitAndSaveIntegration:
    def test_fit_persists_and_load_reads_back(self, tmp_path):
        path = tmp_path / "tr.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in _records(60)) + "\n")
        config = {"forecast_log_path": str(path)}

        cmap = fit_and_save(config)
        assert not cmap.is_empty()
        assert (tmp_path / "tr.jsonl.calibration.json").exists()
        assert load_map(config).calibrate(80, "1h") == pytest.approx(50, abs=5)
