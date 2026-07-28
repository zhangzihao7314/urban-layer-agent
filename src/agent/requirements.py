"""Structured planning requirements used by the elicitation loop."""

from dataclasses import asdict, dataclass, field
from typing import List, Optional

from src.agent.context_parser import parse_planning_context


@dataclass
class PlanningRequirement:
    polygon_id: int
    original_request: str
    planning_typology: Optional[str] = None
    overall_goal: Optional[str] = None
    primary_function: Optional[str] = None
    vegetation_preference: Optional[str] = None
    openness_preference: Optional[str] = None
    water_allowed: Optional[bool] = None
    constraints: List[str] = field(default_factory=list)
    answers: List[str] = field(default_factory=list)
    question_count: int = 0
    assumptions: List[str] = field(default_factory=list)
    asked_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


FUNCTION_KEYWORDS = {
    "cooling": ["cool", "heat", "temperature", "shade", "降温", "热岛"],
    "recreation": ["recreation", "activity", "play", "leisure", "活动", "休闲"],
    "biodiversity": ["biodiversity", "habitat", "ecology", "生态", "生物多样性"],
    "stormwater": ["stormwater", "flood", "drainage", "rain", "雨洪", "排水"],
}


def update_requirement(requirement: PlanningRequirement, text: str) -> PlanningRequirement:
    normalized = text.lower()
    context = parse_planning_context(text)
    requirement.answers.append(text)
    user_uncertain = any(
        phrase in normalized for phrase in [
            "i don't know", "not sure", "you recommend", "either option",
            "不知道", "不确定", "你来推荐",
        ]
    )
    requirement.planning_typology = requirement.planning_typology or context.planning_typology
    for function, keywords in FUNCTION_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            requirement.primary_function = function
            break
    if any(word in normalized for word in ["dense tree", "wooded", "forest", "tree-dominated", "树林", "树冠"]):
        requirement.vegetation_preference = "dense_trees"
    elif any(word in normalized for word in ["some tree", "scattered tree", "partial shade", "部分树", "稀疏树"]):
        requirement.vegetation_preference = "scattered_trees"
    elif any(word in normalized for word in ["grass", "lawn", "low plant", "草地", "草坪"]):
        requirement.vegetation_preference = "low_plants"
    if any(word in normalized for word in [
        "remain open", "keep open", "keep it open", "open space",
        "open grass", "open park", "开放",
    ]):
        requirement.openness_preference = "open"
    elif any(word in normalized for word in ["enclosed", "dense canopy", "封闭", "茂密树冠"]):
        requirement.openness_preference = "enclosed"
    if any(word in normalized for word in ["without water", "no water", "not necessary", "不要水", "不需要水"]):
        requirement.water_allowed = False
        if "exclude:Water" not in requirement.constraints:
            requirement.constraints.append("exclude:Water")
    elif any(word in normalized for word in ["allow water", "include water", "pond", "lake", "允许水", "池塘"]):
        requirement.water_allowed = True
        requirement.constraints = [
            item for item in requirement.constraints if item != "exclude:Water"
        ]
    if user_uncertain:
        requirement.assumptions.append(
            "User was uncertain; use a transparent default and show alternatives."
        )
    return requirement


def requirement_to_description(requirement: PlanningRequirement) -> str:
    parts = [requirement.planning_typology or requirement.original_request]
    mapping = {
        "dense_trees": "wooded dense trees",
        "scattered_trees": "scattered trees with partial shade",
        "low_plants": "grass low plants",
    }
    if requirement.vegetation_preference:
        parts.append(mapping[requirement.vegetation_preference])
    if requirement.openness_preference == "open":
        parts.append("open")
    if requirement.water_allowed is False:
        parts.append("without water")
    return " ".join(parts)
