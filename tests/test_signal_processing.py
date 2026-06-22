"""Tests for the rating/direction heuristics and the SignalProcessor adapter.

The Portfolio Manager now produces a typed ``Forecast`` rendered to markdown
with a ``**Primary signal (next 1h): <Direction>**`` marker. ``parse_direction``
in ``tradingagents.agents.utils.rating`` extracts that headline direction with
no extra LLM call, and SignalProcessor is a thin adapter that delegates to it.
``parse_rating`` (the legacy 5-tier heuristic) is retained and still covered.
"""

import pytest

from tradingagents.agents.utils.rating import (
    RATINGS_5_TIER,
    parse_direction,
    parse_rating,
)
from tradingagents.graph.signal_processing import SignalProcessor

# ---------------------------------------------------------------------------
# Heuristic parser
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseRating:
    def test_explicit_label_buy(self):
        assert parse_rating("Rating: Buy\nReasoning here.") == "Buy"

    def test_explicit_label_overweight(self):
        assert parse_rating("Rating: Overweight\nDetails.") == "Overweight"

    def test_explicit_label_with_markdown_bold_value(self):
        # Regression: Rating: **Sell** — markdown around the value.
        assert parse_rating("Rating: **Sell**\nExit immediately.") == "Sell"

    def test_explicit_label_with_markdown_bold_label(self):
        assert parse_rating("**Rating**: Underweight\nTrim exposure.") == "Underweight"

    def test_rendered_pm_markdown_shape(self):
        # The exact shape produced by render_pm_decision must always parse.
        text = (
            "**Rating**: Buy\n\n"
            "**Executive Summary**: Enter at $189-192, 6% portfolio cap.\n\n"
            "**Investment Thesis**: AI capex cycle intact; institutional flows constructive."
        )
        assert parse_rating(text) == "Buy"

    def test_explicit_label_wins_over_prose_with_markdown(self):
        text = (
            "The buy thesis is weakened by guidance.\n"
            "Rating: **Sell**\n"
            "Exit before earnings."
        )
        assert parse_rating(text) == "Sell"

    def test_no_rating_returns_default(self):
        assert parse_rating("No clear directional signal at this time.") == "Hold"

    def test_no_rating_custom_default(self):
        assert parse_rating("Plain prose.", default="Underweight") == "Underweight"

    def test_all_five_tiers_recognised(self):
        for r in RATINGS_5_TIER:
            assert parse_rating(f"Rating: {r}") == r


# ---------------------------------------------------------------------------
# SignalProcessor: thin adapter over the heuristic
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParseDirection:
    def test_marker_up(self):
        assert parse_direction("**Primary signal (next 1h): Up**") == "Up"

    def test_marker_down(self):
        assert parse_direction("Primary signal (next 1h): Down") == "Down"

    def test_fallback_first_direction_word(self):
        assert parse_direction("The view leans Flat after the bounce.") == "Flat"

    def test_no_direction_returns_default(self):
        assert parse_direction("No clear read at this time.") == "Flat"

    def test_no_direction_custom_default(self):
        assert parse_direction("Plain prose.", default="Down") == "Down"


@pytest.mark.unit
class TestSignalProcessor:
    def test_returns_direction_from_forecast_markdown(self):
        sp = SignalProcessor()
        # A shorter-horizon row with a different direction must not steal the
        # headline: the 1h primary-signal marker is the source of truth.
        md = (
            "**Primary signal (next 1h): Up**\n\n"
            "| Next 5m | Down | $1 | n/a | Low (50%) |\n"
            "| Next 1h | Up | $1 | n/a | Medium (60%) |"
        )
        assert sp.process_signal(md) == "Up"

    def test_makes_no_llm_calls(self):
        """SignalProcessor must not invoke the LLM it was constructed with —
        the direction is parseable from the rendered forecast markdown directly."""
        from unittest.mock import MagicMock

        llm = MagicMock()
        sp = SignalProcessor(llm)
        sp.process_signal("**Primary signal (next 1h): Down**")
        llm.invoke.assert_not_called()
        llm.with_structured_output.assert_not_called()

    def test_default_when_no_signal_present(self):
        sp = SignalProcessor()
        assert sp.process_signal("Plain prose without a directional signal.") == "Flat"
