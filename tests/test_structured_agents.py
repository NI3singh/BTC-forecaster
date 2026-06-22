"""Tests for structured-output agents (Trader, Research Manager, Sentiment Analyst).

The Portfolio Manager has its own coverage in tests/test_memory_log.py
(which exercises the full memory-log → PM injection cycle).  This file
covers the parallel schemas, render functions, and graceful-fallback
behavior we added for the Trader, Research Manager, and Sentiment Analyst
so they share the same deterministic output shape.
"""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from tests._forecast_helpers import make_forecast
from tradingagents.agents.analysts.sentiment_analyst import create_sentiment_analyst
from tradingagents.agents.managers.research_manager import create_research_manager
from tradingagents.agents.schemas import (
    Direction,
    ResearchPlan,
    SentimentBand,
    SentimentReport,
    TraderProposal,
    render_forecast,
    render_research_plan,
    render_sentiment_report,
    render_trader_proposal,
)
from tradingagents.agents.trader.trader import create_trader

# ---------------------------------------------------------------------------
# Render functions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderTraderProposal:
    def test_minimal_required_fields(self):
        p = TraderProposal(direction=Direction.FLAT, reasoning="Balanced setup; no edge.")
        md = render_trader_proposal(p)
        assert "**Direction**: Flat" in md
        assert "**Reasoning**: Balanced setup; no edge." in md
        # Greppable marker of the desk's preliminary directional call.
        assert "FORECAST (next 1-4h): **FLAT**" in md

    def test_optional_key_level_included_when_present(self):
        p = TraderProposal(
            direction=Direction.UP,
            reasoning="Reclaimed VWMA on rising volume.",
            key_level=189.5,
        )
        md = render_trader_proposal(p)
        assert "**Direction**: Up" in md
        assert "**Key Level**: 189.5" in md
        assert "FORECAST (next 1-4h): **UP**" in md

    def test_optional_key_level_omitted_when_absent(self):
        p = TraderProposal(direction=Direction.DOWN, reasoning="Rejected at resistance.")
        md = render_trader_proposal(p)
        assert "Key Level" not in md
        assert "FORECAST (next 1-4h): **DOWN**" in md


@pytest.mark.unit
class TestRenderResearchPlan:
    def test_required_fields(self):
        p = ResearchPlan(
            bias=Direction.UP,
            rationale="Bull case carried; momentum intact.",
            what_to_watch="A 1h close back below VWMA flips the bias.",
        )
        md = render_research_plan(p)
        assert "**Bias**: Up" in md
        assert "**Rationale**: Bull case carried" in md
        assert "**What to Watch**: A 1h close" in md

    def test_all_directions_render(self):
        for direction in Direction:
            p = ResearchPlan(bias=direction, rationale="r", what_to_watch="w")
            md = render_research_plan(p)
            assert f"**Bias**: {direction.value}" in md


@pytest.mark.unit
class TestRenderForecast:
    def _forecast(self, **kw):
        base = {
            "direction_1h": Direction.UP, "expected_price_1h": 65950.0,
            "range_low_1h": 65700.0, "range_high_1h": 66150.0, "confidence_1h": 62,
            "direction_4h": Direction.DOWN, "expected_price_4h": 65400.0,
            "range_low_4h": 64900.0, "range_high_4h": 65900.0, "confidence_4h": 48,
            "reasons": "MACD turned up off oversold; ATR ~250/hr.",
        }
        base.update(kw)
        return make_forecast(**base)

    def test_primary_signal_marker_and_table(self):
        md = render_forecast(self._forecast())
        assert "**Primary signal (next 1h): Up**" in md
        assert "| Next 5m | Up |" in md
        assert "| Next 30m | Up |" in md
        assert "| Next 1h | Up |" in md
        assert "| Next 2h | Up |" in md
        assert "| Next 4h | Down |" in md
        assert "Medium (62%)" in md
        assert "Low (48%)" in md

    def test_optional_levels_and_invalidation(self):
        md = render_forecast(self._forecast(
            key_levels="support 64900 / resistance 66150",
            invalidation="1h close below 64900 flips the 4h view",
        ))
        assert "**Key levels:** support 64900" in md
        assert "**What would invalidate this:**" in md

    def test_confidence_bands(self):
        from tradingagents.agents.schemas import _confidence_band
        assert _confidence_band(75) == "High"
        assert _confidence_band(60) == "Medium"
        assert _confidence_band(40) == "Low"

    def test_forecast_anchor_header(self):
        from tradingagents.agents.schemas import render_forecast_anchor
        h = render_forecast_anchor("BTC-USD", "2026-06-18T09:50:00", 64200.98)
        assert "BTC-USD" in h
        assert "2026-06-18 09:50 UTC" in h  # ISO timestamp formatted for display
        assert "$64,200.98" in h  # real spot price at forecast time


# ---------------------------------------------------------------------------
# Trader agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_trader_state():
    return {
        "company_of_interest": "NVDA",
        "investment_plan": "**Bias**: Up\n**Rationale**: ...\n**What to Watch**: ...",
    }


def _structured_trader_llm(captured: dict, proposal: TraderProposal | None = None):
    """Build a MagicMock LLM whose with_structured_output binding captures the
    prompt and returns a real TraderProposal so render_trader_proposal works.
    """
    if proposal is None:
        proposal = TraderProposal(
            direction=Direction.UP,
            reasoning="Strong setup.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or proposal
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestTraderAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        proposal = TraderProposal(
            direction=Direction.UP,
            reasoning="AI capex cycle intact; institutional flows constructive.",
            key_level=189.5,
        )
        llm = _structured_trader_llm(captured, proposal)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        plan = result["trader_investment_plan"]
        assert "**Direction**: Up" in plan
        assert "**Key Level**: 189.5" in plan
        assert "FORECAST (next 1-4h): **UP**" in plan
        # The same rendered markdown is also added to messages for downstream agents.
        assert plan in result["messages"][0].content

    def test_prompt_includes_research_verdict(self):
        captured = {}
        llm = _structured_trader_llm(captured)
        trader = create_trader(llm)
        trader(_make_trader_state())
        # The research manager's verdict is in the user message of the captured prompt.
        prompt = captured["prompt"]
        assert any("Research Manager's verdict" in m["content"] for m in prompt)

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = (
            "**Direction**: Down\n\nMomentum fading into resistance.\n\n"
            "FORECAST (next 1-4h): **DOWN**"
        )
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        trader = create_trader(llm)
        result = trader(_make_trader_state())
        assert result["trader_investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Research Manager agent: structured happy path + fallback
# ---------------------------------------------------------------------------


def _make_rm_state():
    return {
        "company_of_interest": "NVDA",
        "investment_debate_state": {
            "history": "Bull and bear arguments here.",
            "bull_history": "Bull says...",
            "bear_history": "Bear says...",
            "current_response": "",
            "judge_decision": "",
            "count": 1,
        },
    }


def _structured_rm_llm(captured: dict, plan: ResearchPlan | None = None):
    if plan is None:
        plan = ResearchPlan(
            bias=Direction.FLAT,
            rationale="Balanced view across both sides.",
            what_to_watch="Reassess if price breaks the overnight range.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or plan
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestResearchManagerAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        plan = ResearchPlan(
            bias=Direction.UP,
            rationale="Bull case is stronger; momentum intact.",
            what_to_watch="Watch the 1h VWMA reclaim holding.",
        )
        llm = _structured_rm_llm(captured, plan)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        ip = result["investment_plan"]
        assert "**Bias**: Up" in ip
        assert "**Rationale**: Bull case" in ip
        assert "**What to Watch**: Watch the 1h" in ip

    def test_prompt_uses_directional_scale(self):
        """The RM prompt must list the Up/Flat/Down directional options."""
        captured = {}
        llm = _structured_rm_llm(captured)
        rm = create_research_manager(llm)
        rm(_make_rm_state())
        prompt = captured["prompt"]
        for d in ("Up", "Flat", "Down"):
            assert f"**{d}**" in prompt, f"missing {d} in prompt"

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain_response = "**Bias**: Down\n\n**Rationale**: ...\n\n**What to Watch**: ..."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain_response)
        rm = create_research_manager(llm)
        result = rm(_make_rm_state())
        assert result["investment_plan"] == plain_response


# ---------------------------------------------------------------------------
# Sentiment Analyst: schema, render, structured happy path + fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRenderSentimentReport:
    def test_header_contains_band_and_score(self):
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH,
            overall_score=7.2,
            confidence="high",
            narrative="Source breakdown here.",
        )
        md = render_sentiment_report(report)
        assert "**Overall Sentiment:** **Bullish**" in md
        assert "(Score: 7.2/10)" in md

    def test_header_contains_confidence(self):
        report = SentimentReport(
            overall_band=SentimentBand.NEUTRAL,
            overall_score=5.0,
            confidence="low",
            narrative="Limited data.",
        )
        assert "**Confidence:** Low" in render_sentiment_report(report)

    def test_narrative_preserved_in_output(self):
        narrative = "## Breakdown\n\nStockTwits: 70% bullish.\n\n| Signal | Direction |\n|---|---|\n| News | Neutral |"
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BULLISH,
            overall_score=6.0,
            confidence="medium",
            narrative=narrative,
        )
        assert narrative in render_sentiment_report(report)

    def test_all_six_bands_render(self):
        for band in SentimentBand:
            report = SentimentReport(
                overall_band=band, overall_score=5.0,
                confidence="medium", narrative="n",
            )
            assert band.value in render_sentiment_report(report)

    def test_score_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            SentimentReport(
                overall_band=SentimentBand.BULLISH, overall_score=11.0,
                confidence="high", narrative="n",
            )


def _make_sentiment_state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-01-15",
        "asset_type": "stock",
        "messages": [],
    }


def _structured_sentiment_llm(captured: dict, report: SentimentReport | None = None):
    """MagicMock LLM whose structured binding captures the prompt and returns
    a real SentimentReport so render_sentiment_report works."""
    if report is None:
        report = SentimentReport(
            overall_band=SentimentBand.BULLISH, overall_score=7.5,
            confidence="high",
            narrative="StockTwits 75% bullish. News constructive. Reddit upbeat.",
        )
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) or report
    )
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
class TestSentimentAnalystAgent:
    def test_structured_path_produces_rendered_markdown(self):
        captured = {}
        report = SentimentReport(
            overall_band=SentimentBand.MILDLY_BEARISH, overall_score=4.0,
            confidence="medium", narrative="Mixed signals across sources.",
        )
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured, report))
        sr = analyst(_make_sentiment_state())["sentiment_report"]
        assert "**Overall Sentiment:** **Mildly Bearish**" in sr
        assert "(Score: 4.0/10)" in sr
        assert "Mixed signals across sources." in sr

    def test_sentiment_report_also_in_messages(self):
        captured = {}
        analyst = create_sentiment_analyst(_structured_sentiment_llm(captured))
        result = analyst(_make_sentiment_state())
        assert len(result["messages"]) == 1
        assert result["sentiment_report"] == result["messages"][0].content

    def test_prompt_contains_ticker(self):
        captured = {}
        create_sentiment_analyst(_structured_sentiment_llm(captured))(_make_sentiment_state())
        assert any("NVDA" in str(m) for m in captured["prompt"])

    def test_falls_back_to_freetext_when_structured_unavailable(self):
        plain = "**Overall Sentiment:** **Bearish** (Score: 3.0/10)\n**Confidence:** Low\n\nLimited data."
        llm = MagicMock()
        llm.with_structured_output.side_effect = NotImplementedError("provider unsupported")
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain

    def test_falls_back_to_freetext_when_structured_call_fails(self):
        plain = "Fallback free-text sentiment."
        structured = MagicMock()
        structured.invoke.side_effect = ValueError("bad JSON from model")
        llm = MagicMock()
        llm.with_structured_output.return_value = structured
        llm.invoke.return_value = MagicMock(content=plain)
        assert create_sentiment_analyst(llm)(_make_sentiment_state())["sentiment_report"] == plain
