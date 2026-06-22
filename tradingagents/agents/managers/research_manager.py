"""Research Manager: turns the bull/bear debate into a structured investment plan for the trader."""

from __future__ import annotations

from tradingagents.agents.schemas import ResearchPlan, render_research_plan
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.agents.utils.structured import (
    bind_structured,
    invoke_structured_or_freetext,
)


def create_research_manager(llm):
    structured_llm = bind_structured(llm, ResearchPlan, "Research Manager")

    def research_manager_node(state) -> dict:
        instrument_context = get_instrument_context_from_state(state)
        history = state["investment_debate_state"].get("history", "")

        investment_debate_state = state["investment_debate_state"]

        prompt = f"""As the Research Manager and debate facilitator on an intraday price-forecasting desk, critically evaluate this round of the bull/bear debate and deliver a clear directional verdict for the next 5 minutes to 4 hours that the trader can act on.

{instrument_context}

---

**Directional bias** (choose exactly one):
- **Up**: The bull case is stronger; price is more likely to rise over the next 5 minutes to 4 hours.
- **Flat**: The evidence is genuinely balanced, or price is likely to stay inside the noise band.
- **Down**: The bear case is stronger; price is more likely to fall over the next 5 minutes to 4 hours.

Commit to Up or Down whenever the debate's strongest arguments warrant it; reserve Flat for a genuinely balanced setup. Ground the verdict in the intraday technical, news, and sentiment evidence, and flag the levels that would confirm or flip it.

---

**Debate History:**
{history}""" + get_language_instruction()

        investment_plan = invoke_structured_or_freetext(
            structured_llm,
            llm,
            prompt,
            render_research_plan,
            "Research Manager",
        )

        new_investment_debate_state = {
            "judge_decision": investment_plan,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": investment_plan,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": investment_plan,
        }

    return research_manager_node
