"""Trader: turns the Research Manager's investment plan into a concrete transaction proposal."""

from __future__ import annotations

import functools

from langchain_core.messages import AIMessage

from tradingagents.agents.schemas import TraderProposal, render_trader_proposal
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_trader(llm):
    structured_llm = bind_structured(llm, TraderProposal, "Trader")

    def trader_node(state, name):
        company_name = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        investment_plan = state["investment_plan"]

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a trader on an intraday price-forecasting desk. Based on the "
                    "analysts' intraday reports and the research manager's directional verdict, "
                    "commit to a preliminary directional call (Up, Flat, or Down) for the next "
                    "1-4 hours, with concise reasoning and the key intraday level that matters. "
                    "This is a directional forecast, not investment advice or position sizing."
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Here is the research manager's directional verdict for {company_name}. "
                    f"{instrument_context} It synthesizes the intraday technical, news, and "
                    f"sentiment analysis. Use it as the basis for your preliminary 1-4 hour "
                    f"directional call.\n\nResearch Manager's verdict: {investment_plan}\n\n"
                    f"Commit to Up, Flat, or Down and explain why."
                ),
            },
        ]

        trader_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            messages,
            render_trader_proposal,
            "Trader",
        )

        return {
            "messages": [AIMessage(content=trader_plan)],
            "trader_investment_plan": trader_plan,
            "sender": name,
        }

    return functools.partial(trader_node, name="Trader")
