"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


class Direction(str, Enum):
    """3-tier directional outcome used by the intraday forecasting agents.

    Replaces the investor Buy/Hold/Sell framing for the forecasting pipeline:
    the question is which way price moves over the next 1-4 hours, not whether
    to take a position. ``Flat`` means inside the fee/noise band (no clear move).
    """

    UP = "Up"
    FLAT = "Flat"
    DOWN = "Down"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured directional verdict produced by the Research Manager.

    Hand-off to the Trader on the intraday forecasting desk: ``bias`` pins the
    near-term directional view, ``rationale`` captures which side of the
    bull/bear debate carried the argument, and ``what_to_watch`` flags the
    concrete intraday levels/signals that would confirm or flip it.
    """

    bias: Direction = Field(
        description=(
            "The directional bias for the next 1-4 hours. Exactly one of Up / "
            "Flat / Down. Reserve Flat for when the debate is genuinely balanced "
            "or price is likely to stay inside the noise band; otherwise commit "
            "to the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments drove the directional call. "
            "Speak naturally, as if to a teammate."
        ),
    )
    what_to_watch: str = Field(
        description=(
            "Concrete intraday levels, signals, or imminent events the trader "
            "should watch over the next 1-4 hours, including what would flip the "
            "bias."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Bias**: {plan.bias.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**What to Watch**: {plan.what_to_watch}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured preliminary directional call produced by the Trader.

    On the intraday forecasting desk the trader reads the Research Manager's
    directional verdict and the analyst reports, then commits to a preliminary
    call for the next 1-4 hours: which way, why, and the key intraday level
    that matters.
    """

    direction: Direction = Field(
        description="The preliminary directional call for the next 1-4 hours. Exactly one of Up / Flat / Down.",
    )
    reasoning: str = Field(
        description=(
            "The case for this call, anchored in the analysts' intraday reports "
            "and the research plan. Two to four sentences."
        ),
    )
    key_level: float | None = Field(
        default=None,
        description=(
            "Optional key intraday price level (support, resistance, or trigger) "
            "in the instrument's quote currency."
        ),
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FORECAST (next 1-4h): **UP/FLAT/DOWN**`` line is a stable,
    greppable marker of the desk's preliminary directional call.
    """
    parts = [
        f"**Direction**: {proposal.direction.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.key_level is not None:
        parts.extend(["", f"**Key Level**: {proposal.key_level}"])
    parts.extend([
        "",
        f"FORECAST (next 1-4h): **{proposal.direction.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: PortfolioRating = Field(
        description=(
            "The final position rating. Exactly one of Buy / Overweight / Hold / "
            "Underweight / Sell, picked based on the analysts' debate."
        ),
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: float | None = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    time_horizon: str | None = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision back to the markdown shape the rest of the system expects.

    Memory log, CLI display, and saved report files all read this markdown,
    so the rendered output preserves the exact section headers (``**Rating**``,
    ``**Executive Summary**``, ``**Investment Thesis**``) that downstream
    parsers and the report writers already handle.
    """
    parts = [
        f"**Rating**: {decision.rating.value}",
        "",
        f"**Executive Summary**: {decision.executive_summary}",
        "",
        f"**Investment Thesis**: {decision.investment_thesis}",
    ]
    if decision.price_target is not None:
        parts.extend(["", f"**Price Target**: {decision.price_target}"])
    if decision.time_horizon:
        parts.extend(["", f"**Time Horizon**: {decision.time_horizon}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Forecast (final intraday deliverable)
# ---------------------------------------------------------------------------


class Forecast(BaseModel):
    """The final intraday price forecast produced by the Portfolio Manager.

    Replaces the investor PortfolioDecision for the forecasting pipeline: a
    per-horizon (next 1h, next 4h) directional call with an approximate price,
    an expected range, and a confidence, plus the reasons and the levels that
    would confirm or invalidate it. The numbers are the desk's best estimate;
    the Stage-5 track-record loop keeps the confidence honest over time.
    """

    direction_1h: Direction = Field(description="Direction for the NEXT 1 HOUR: Up / Flat / Down.")
    expected_price_1h: float = Field(description="Approximate expected price at the end of the next 1 hour, in the quote currency.")
    range_low_1h: float | None = Field(default=None, description="Low end of the expected price range over the next 1 hour.")
    range_high_1h: float | None = Field(default=None, description="High end of the expected price range over the next 1 hour.")
    confidence_1h: int = Field(ge=0, le=100, description="Confidence (0-100) in the next-1-hour direction call.")

    direction_4h: Direction = Field(description="Direction for the NEXT 4 HOURS: Up / Flat / Down.")
    expected_price_4h: float = Field(description="Approximate expected price at the end of the next 4 hours, in the quote currency.")
    range_low_4h: float | None = Field(default=None, description="Low end of the expected price range over the next 4 hours.")
    range_high_4h: float | None = Field(default=None, description="High end of the expected price range over the next 4 hours.")
    confidence_4h: int = Field(ge=0, le=100, description="Confidence (0-100) in the next-4-hour direction call.")

    reasons: str = Field(
        description=(
            "2-4 sentences citing the actual drivers behind the calls: momentum/"
            "MACD, ATR-implied range, key intraday levels reclaimed or lost, and "
            "any breaking news/sentiment that shaped the direction or widened the "
            "range."
        ),
    )
    key_levels: str | None = Field(
        default=None,
        description="Key intraday support/resistance levels, e.g. 'support $64,900 - resistance $66,150'.",
    )
    invalidation: str | None = Field(
        default=None,
        description="What price action would invalidate this forecast, e.g. 'a 1h close below $64,900 flips the 4h view'.",
    )


def _confidence_band(pct: int) -> str:
    if pct >= 70:
        return "High"
    if pct >= 55:
        return "Medium"
    return "Low"


def _fmt_range(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return "n/a"
    return f"${low:,.0f} - ${high:,.0f}"


def render_forecast(f: Forecast) -> str:
    """Render a Forecast to the markdown the CLI, reports, and signal parser consume.

    The leading ``Primary signal (next 1h)`` line is a stable, greppable marker
    the signal processor reads to extract the headline direction.
    """
    parts = [
        f"**Primary signal (next 1h): {f.direction_1h.value}**",
        "",
        "| Horizon | Direction | Approx. price | Expected range | Confidence |",
        "| --- | --- | --- | --- | --- |",
        f"| Next 1h | {f.direction_1h.value} | ${f.expected_price_1h:,.2f} | "
        f"{_fmt_range(f.range_low_1h, f.range_high_1h)} | "
        f"{_confidence_band(f.confidence_1h)} ({f.confidence_1h}%) |",
        f"| Next 4h | {f.direction_4h.value} | ${f.expected_price_4h:,.2f} | "
        f"{_fmt_range(f.range_low_4h, f.range_high_4h)} | "
        f"{_confidence_band(f.confidence_4h)} ({f.confidence_4h}%) |",
        "",
        f"**Why:** {f.reasons}",
    ]
    if f.key_levels:
        parts.extend(["", f"**Key levels:** {f.key_levels}"])
    if f.invalidation:
        parts.extend(["", f"**What would invalidate this:** {f.invalidation}"])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Sentiment Analyst
# ---------------------------------------------------------------------------


class SentimentBand(str, Enum):
    """Discrete sentiment direction produced by the Sentiment Analyst.

    Six tiers keep the signal granular enough to be actionable while remaining
    small enough for every provider to map reliably from its JSON output.
    """

    BULLISH = "Bullish"
    MILDLY_BULLISH = "Mildly Bullish"
    NEUTRAL = "Neutral"
    MIXED = "Mixed"
    MILDLY_BEARISH = "Mildly Bearish"
    BEARISH = "Bearish"


class SentimentReport(BaseModel):
    """Structured sentiment report produced by the Sentiment Analyst.

    Replaces the previous free-form prose output so downstream consumers
    (dashboards, audit logs, PDF renderers, other agents) can read
    ``overall_band`` and ``overall_score`` without maintaining fragile regex
    fallbacks that drift with every model release. ``narrative`` preserves the
    rich source-by-source analysis; ``render_sentiment_report`` prepends a
    deterministic header so the saved report stays human-readable.
    """

    overall_band: SentimentBand = Field(
        description=(
            "Overall sentiment direction. Exactly one of: "
            "Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. "
            "Use Mixed when sources point in clearly different directions. "
            "Use Neutral only when all sources are genuinely silent or non-committal."
        ),
    )
    overall_score: float = Field(
        ge=0.0,
        le=10.0,
        description=(
            "Numeric sentiment intensity on a 0–10 scale. "
            "0 = maximally bearish, 5 = neutral, 10 = maximally bullish. "
            "Guideline for consistency with overall_band: "
            "Bullish ~6.5–10, Mildly Bullish ~5.5–6.4, Neutral/Mixed ~4.5–5.5, "
            "Mildly Bearish ~3.5–4.4, Bearish ~0–3.4. "
            "Only the 0–10 bounds are enforced."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Confidence in the assessment based on data quality and sample size. "
            "Use 'low' when one or more sources returned a placeholder or fewer "
            "than 5 data points; 'medium' when data is present but sparse; "
            "'high' when all three sources returned substantive data."
        ),
    )
    narrative: str = Field(
        description=(
            "Full sentiment report covering, in order: "
            "(1) source-by-source breakdown with specific evidence (cite message "
            "counts, ratios, notable posts); "
            "(2) cross-source divergences and alignments; "
            "(3) dominant narrative themes; "
            "(4) catalysts and risks surfaced by the data; "
            "(5) a markdown table summarising key sentiment signals, their "
            "direction, source, and supporting evidence. "
            "Keep it informative and substantive: develop each section thoroughly "
            "with concrete evidence so every point adds new signal for the trader."
        ),
    )


def render_sentiment_report(report: SentimentReport) -> str:
    """Render a SentimentReport to the markdown shape the rest of the system expects.

    The structured header (band + score + confidence) is prepended to the
    narrative so the saved report is both human-readable and machine-parseable
    without regex.
    """
    return "\n".join([
        f"**Overall Sentiment:** **{report.overall_band.value}** "
        f"(Score: {report.overall_score:.1f}/10)",
        f"**Confidence:** {report.confidence.capitalize()}",
        "",
        report.narrative,
    ])
