from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)


def create_bear_researcher(llm):
    def bear_node(state) -> dict:
        investment_debate_state = state["investment_debate_state"]
        history = investment_debate_state.get("history", "")
        bear_history = investment_debate_state.get("bear_history", "")

        current_response = investment_debate_state.get("current_response", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]
        instrument_context = get_instrument_context_from_state(state)
        asset_type = state.get("asset_type", "stock")
        target_label = "stock" if asset_type == "stock" else "asset"
        fundamentals_label = (
            "Company fundamentals report"
            if asset_type == "stock"
            else "Asset fundamentals report (may be unavailable for crypto)"
        )

        prompt = f"""You are a Bear Analyst on an intraday price-forecasting desk. Build the strongest evidence-based case that the {target_label}'s price will FALL over the NEXT 5 MINUTES TO 4 HOURS (across the 5m, 15m, 30m, 1h, 2h and 4h horizons). Argue the short-horizon bear view and counter the bull's points directly using the intraday (5-minute) evidence.

Key points to focus on:

- Momentum & trend: bearish MACD/RSI shifts, rejection at short-term moving averages, lower highs on the intraday (5-minute) chart, overbought exhaustion.
- Breakdown / rejection: a key intraday support about to give way, a failed breakout, weak or declining volume (VWMA) behind any bounce.
- Catalysts: breaking news or a sentiment shift that could push price down within hours.
- Bull counterpoints: dismantle the bull's case with specific intraday data and concrete levels, exposing over-optimistic assumptions.
- Engagement: conversational, engaging directly with the bull analyst — debate, don't just list facts.

Resources available:

{instrument_context}
Market (technical) report: {market_research_report}
Social media sentiment report: {sentiment_report}
Latest world affairs news: {news_report}
{fundamentals_label}: {fundamentals_report}
Conversation history of the debate: {history}
Last bull argument: {current_response}
Deliver a compelling short-horizon (5m-4h) bear case, refute the bull's claims, and engage in a dynamic debate.
""" + get_language_instruction()

        response = llm.invoke(prompt)

        argument = f"Bear Analyst: {response.content}"

        new_investment_debate_state = {
            "history": history + "\n" + argument,
            "bear_history": bear_history + "\n" + argument,
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": argument,
            "count": investment_debate_state["count"] + 1,
        }

        return {"investment_debate_state": new_investment_debate_state}

    return bear_node
