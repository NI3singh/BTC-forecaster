"""Extract the 5-tier portfolio rating from the Portfolio Manager's decision.

The Portfolio Manager produces a typed ``PortfolioDecision`` via structured
output and renders it to markdown that always carries a ``**Rating**: X``
header (see :func:`tradingagents.agents.schemas.render_pm_decision`).  The
deterministic heuristic in :mod:`tradingagents.agents.utils.rating` is more
than sufficient to extract that rating; no extra LLM call is needed.

This module exists for backwards compatibility with callers that expect a
``SignalProcessor.process_signal(text)`` interface.
"""

from __future__ import annotations

from typing import Any

from tradingagents.agents.utils.rating import parse_direction


class SignalProcessor:
    """Read the headline directional signal out of a Portfolio Manager forecast."""

    def __init__(self, quick_thinking_llm: Any = None):
        # The LLM argument is accepted for backwards compatibility but no
        # longer used: the PM's structured Forecast guarantees the direction is
        # parseable from the rendered markdown without a second LLM call.
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """Return the headline next-1h direction: Up / Flat / Down."""
        return parse_direction(full_signal)
