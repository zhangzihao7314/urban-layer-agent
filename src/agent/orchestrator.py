"""Small, deterministic Agent loop for choosing and executing domain tools.

The LLM may supply natural-language intent, but authoritative facts always come
from tools backed by the professor-provided tables.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List

from src.agent.intent_router import route_intent
from src.agent.tools import UrbanLayerTools


@dataclass
class AgentRun:
    intent: str
    tool_calls: List[Dict[str, Any]]
    result: Dict[str, Any]
    needs_user_confirmation: bool = False


class UrbanLayerAgent:
    def __init__(self, tools: UrbanLayerTools):
        self.tools = tools

    def run(self, user_text: str, llm_intent: str | None = None) -> AgentRun:
        decision = route_intent(user_text, llm_intent)
        calls = []
        if decision.intent.value == "set_goal":
            result = self.tools.recommend_goal_types(user_text)
            calls.append({"tool": "recommend_goal_types", "arguments": {"goal": user_text}})
            return AgentRun(decision.intent.value, calls, result)
        if decision.intent.value in {"set_polygon", "apply_all"}:
            description = decision.target_description or user_text
            search = self.tools.search_planning_typology(description)
            calls.append({"tool": "search_planning_typology", "arguments": {"text": description}})
            ranked = self.tools.rank_lcz_candidates(description)
            calls.append({"tool": "rank_lcz_candidates", "arguments": {"text": description}})
            if ranked:
                result = {"context": search["context"], "ranked_candidates": ranked}
                return AgentRun(decision.intent.value, calls, result, needs_user_confirmation=True)
            return AgentRun(decision.intent.value, calls, search, needs_user_confirmation=True)
        return AgentRun(decision.intent.value, calls, {"message": "No domain tool was required."})
