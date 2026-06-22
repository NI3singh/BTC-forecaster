"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import functools

from tradingagents.agents.schemas import Forecast, render_forecast, render_forecast_anchor
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, Forecast, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        company_name = state["company_of_interest"]
        trade_date = state.get("trade_date")

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        # Deterministic market anchor (best-effort): the real spot price, realized
        # volatility, and recent swing levels, fetched once and reused for the
        # prompt anchor block, the vol-scaled range override, and the baseline
        # header. The prompt has always told the model to anchor on the spot price
        # and ATR but never actually supplied them; this closes that gap.
        anchor = None
        if trade_date:
            # Lazy import to avoid an agents <-> forecasting import cycle.
            from tradingagents.forecasting.track_record import intraday_market_anchor
            anchor = intraday_market_anchor(company_name, trade_date)

        anchor_block = ""
        post_process = None
        if anchor:
            from tradingagents.forecasting.ranges import (
                apply_vol_scaled_ranges,
                render_anchor_block,
            )
            anchor_block = render_anchor_block(company_name, anchor) + "\n\n"
            sigma_bar = anchor.get("sigma_bar")
            if sigma_bar:
                post_process = functools.partial(apply_vol_scaled_ranges, sigma_bar=sigma_bar)

        prompt = f"""As the Portfolio Manager on an intraday price-forecasting desk, synthesize the risk analysts' debate and the desk's analysis into the FINAL price forecast for {company_name}, covering all six horizons: 5m, 15m, 30m, 1h, 2h and 4h.

{instrument_context}

{anchor_block}For EACH of the six horizons (5m, 15m, 30m, 1h, 2h, 4h) provide:
- a direction (Up / Flat / Down),
- an approximate expected price in the quote currency,
- an expected price range (low and high), sized from the intraday ATR / volatility and widening with the horizon (these are refined from realized volatility after you answer, so concentrate on a well-centered expected price),
- a confidence from 0-100 — be honest: reserve high confidence for genuinely strong setups; a near-coin-flip is ~50.

Then give the reasons (cite the concrete drivers: momentum/MACD, ATR-implied range, key levels reclaimed or lost, breaking news/sentiment), the key intraday support/resistance levels, and what price action would invalidate the forecast. Anchor your expected prices on the verified spot price above.

**Context:**
- Research Manager's directional verdict: **{research_plan}**
- Trader's preliminary call: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Ground every number in the analysts' evidence; do not fabricate precision.{get_language_instruction()}"""

        final_trade_decision = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_forecast,
            "Portfolio Manager",
            post_process=post_process,
        )

        # Pin the forecast to its real baseline (timestamp + spot price at forecast
        # time) so the output is self-documenting, reusing the anchor fetched above.
        # Best-effort; never blocks the forecast if the data layer is unavailable.
        if anchor:
            final_trade_decision = (
                render_forecast_anchor(company_name, anchor["as_of_iso"], anchor["spot"])
                + "\n\n"
                + final_trade_decision
            )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
        }

    return portfolio_manager_node
