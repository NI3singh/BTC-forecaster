"""Tests for deterministic quant<->agent forecast fusion (pure, no network)."""

import pytest

from tests._forecast_helpers import make_forecast
from tradingagents.agents.schemas import Direction
from tradingagents.forecasting.fusion import (
    _direction_from_p,
    agent_implied_p_up,
    fuse_forecast,
    render_fusion_block,
)


@pytest.mark.unit
class TestImpliedProbAndDirection:
    def test_agent_implied_p_up(self):
        assert agent_implied_p_up(Direction.UP, 70) == pytest.approx(0.70)
        assert agent_implied_p_up(Direction.DOWN, 70) == pytest.approx(0.30)
        assert agent_implied_p_up(Direction.FLAT, 99) == 0.5  # Flat = no info

    def test_direction_from_p(self):
        assert _direction_from_p(0.70) == Direction.UP
        assert _direction_from_p(0.30) == Direction.DOWN
        assert _direction_from_p(0.50) == Direction.FLAT
        assert _direction_from_p(0.51) == Direction.FLAT  # inside flat band


@pytest.mark.unit
class TestFuseForecast:
    def _row(self, sbs, horizon):
        return next(r for r in sbs if r["horizon"] == horizon)

    def test_agreement_blends_confidence(self):
        f = make_forecast(direction_1h=Direction.UP, confidence_1h=60)
        qp = {"1h": {"prob_up": 0.8, "direction": "Up", "confidence": 80}}
        fused, sbs = fuse_forecast(f, qp, weight=0.6)
        assert fused.direction_1h == Direction.UP
        assert fused.confidence_1h == 72  # round(100*(0.6*0.8 + 0.4*0.6))
        assert self._row(sbs, "1h")["disagree"] is False

    def test_disagreement_flips_to_quant_and_trims_conviction(self):
        f = make_forecast(direction_1h=Direction.DOWN, confidence_1h=60)
        qp = {"1h": {"prob_up": 0.8, "direction": "Up", "confidence": 80}}
        fused, sbs = fuse_forecast(f, qp, weight=0.6)
        # fused P = 0.6*0.8 + 0.4*0.4 = 0.64 -> Up; conf 64 capped to 55 on disagreement
        assert fused.direction_1h == Direction.UP
        assert fused.confidence_1h == 55
        assert self._row(sbs, "1h")["disagree"] is True

    def test_horizon_without_quant_is_untouched(self):
        f = make_forecast(direction_4h=Direction.DOWN, confidence_4h=66)
        qp = {"1h": {"prob_up": 0.8, "direction": "Up", "confidence": 80}}  # no 4h
        fused, sbs = fuse_forecast(f, qp)
        assert fused.direction_4h == Direction.DOWN
        assert fused.confidence_4h == 66
        row = self._row(sbs, "4h")
        assert row["quant_dir"] is None and row["fused_dir"] == "Down"

    def test_weight_one_is_quant_only(self):
        f = make_forecast(direction_1h=Direction.DOWN, confidence_1h=90)
        qp = {"1h": {"prob_up": 0.62, "direction": "Up", "confidence": 62}}
        fused, _ = fuse_forecast(f, qp, weight=1.0)
        assert fused.direction_1h == Direction.UP  # ignores the desk entirely


@pytest.mark.unit
class TestRenderFusionBlock:
    def test_renders_table_with_flags(self):
        sbs = [
            {"horizon": "1h", "quant_dir": "Up", "quant_p": 0.62, "agent_dir": "Down",
             "agent_conf": 60, "fused_dir": "Up", "fused_conf": 55, "disagree": True},
            {"horizon": "4h", "quant_dir": None, "quant_p": None, "agent_dir": "Up",
             "agent_conf": 55, "fused_dir": "Up", "fused_conf": 55, "disagree": False},
        ]
        block = render_fusion_block(sbs)
        assert "Quant vs Desk vs Fused" in block
        assert "P=0.62" in block
        assert "⚠" in block
        assert "n/a" in block  # 4h has no quant prediction

    def test_empty_is_blank(self):
        assert render_fusion_block([]) == ""
