"""Validated LLM extraction for flexible requirement answers.

The local LLM extracts language only. It is not allowed to provide LCZ predictor
values or perform GIS mutations.
"""

import json
from typing import Callable, Dict, Optional

from src.agent.requirements import PlanningRequirement


ALLOWED_VALUES = {
    "primary_function": {"cooling", "recreation", "biodiversity", "stormwater", None},
    "vegetation_preference": {"dense_trees", "scattered_trees", "low_plants", None},
    "openness_preference": {"open", "enclosed", None},
    "water_allowed": {True, False, None},
}


def build_extraction_prompt(requirement: PlanningRequirement, user_answer: str) -> str:
    return f"""
You extract planning requirements for one polygon.
Return JSON only. Do not output predictor values, LCZ numbers, or GIS operations.

Current requirement:
{json.dumps(requirement.to_dict(), ensure_ascii=False)}

User answer:
{user_answer}

Allowed output keys and values:
- primary_function: cooling | recreation | biodiversity | stormwater | null
- vegetation_preference: dense_trees | scattered_trees | low_plants | null
- openness_preference: open | enclosed | null
- water_allowed: true | false | null
- user_uncertain: boolean
- confidence: number from 0 to 1

Return only one JSON object.
""".strip()


def parse_extraction(raw: str) -> Optional[Dict]:
    try:
        start, end = raw.find("{"), raw.rfind("}") + 1
        data = json.loads(raw[start:end])
        if not isinstance(data, dict):
            return None
        allowed_keys = set(ALLOWED_VALUES) | {"user_uncertain", "confidence"}
        if set(data) - allowed_keys:
            return None
        for key, allowed in ALLOWED_VALUES.items():
            if key in data and data[key] not in allowed:
                return None
        confidence = float(data.get("confidence", 0.0))
        if not 0.0 <= confidence <= 1.0:
            return None
        data["confidence"] = confidence
        data["user_uncertain"] = bool(data.get("user_uncertain", False))
        return data
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def apply_extraction(requirement: PlanningRequirement, extraction: Dict) -> PlanningRequirement:
    for field_name in ALLOWED_VALUES:
        value = extraction.get(field_name)
        if value is not None:
            setattr(requirement, field_name, value)
    if extraction.get("water_allowed") is False and "exclude:Water" not in requirement.constraints:
        requirement.constraints.append("exclude:Water")
    if extraction.get("water_allowed") is True:
        requirement.constraints = [
            item for item in requirement.constraints if item != "exclude:Water"
        ]
    if extraction.get("user_uncertain"):
        requirement.assumptions.append(
            "The user expressed uncertainty; the Agent will propose alternatives rather than force another answer."
        )
    return requirement


def extract_with_fallback(
    requirement: PlanningRequirement,
    user_answer: str,
    llm_call: Callable[[str], str],
) -> Optional[Dict]:
    try:
        raw = llm_call(build_extraction_prompt(requirement, user_answer))
    except Exception:
        return None
    return parse_extraction(raw or "")


def build_dynamic_question_prompt(requirement: PlanningRequirement, topic: str) -> str:
    return f"""
Write one short, beginner-friendly clarification question for an urban-planning
user. Ask only about: {topic}. Do not mention LCZ codes or predictor names.
Do not ask anything already answered in this requirement:
{json.dumps(requirement.to_dict(), ensure_ascii=False)}
Return the question only, maximum 35 words.
""".strip()
