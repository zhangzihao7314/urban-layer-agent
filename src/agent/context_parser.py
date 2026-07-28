"""Extract structured planning context from ordinary user language."""

from dataclasses import dataclass, field
import re
from typing import List, Optional

from src.agent.knowledge_base import TypologyKnowledgeBase, get_knowledge_base


@dataclass
class PlanningIntent:
    original_text: str
    polygon_id: Optional[int] = None
    planning_typology: Optional[str] = None
    explicit_lcz_type: Optional[str] = None
    preferences: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


PREFERENCE_TERMS = {
    "trees": ["tree", "trees", "forest", "wooded", "canopy"],
    "grass": ["grass", "lawn", "low plant"],
    "water": ["water", "pond", "lake", "canal", "blue"],
    "open": ["open", "ventilation", "permeable"],
    "paved": ["paved", "parking", "road", "square"],
}


def parse_planning_context(text: str, kb: TypologyKnowledgeBase | None = None) -> PlanningIntent:
    kb = kb or get_knowledge_base()
    normalized = text.lower()
    polygon_match = re.search(r"(?:polygon|zone|area)\s*(\d+)", normalized)
    negated_water = any(
        phrase in normalized for phrase in ["without water", "no water", "do not use water", "不要水体"]
    )
    explicit = next((
        name for name in kb.urban_types
        if name.lower() in normalized and not (name == "Water" and negated_water)
    ), None)
    preferences = [
        label for label, terms in PREFERENCE_TERMS.items()
        if any(term in normalized for term in terms)
        and not (label == "water" and negated_water)
    ]
    constraints = []
    if any(term in normalized for term in ["must keep", "preserve", "retain"]):
        constraints.append("preserve_existing")
    if any(term in normalized for term in ["cannot", "must not", "avoid"]):
        constraints.append("negative_constraint")
    return PlanningIntent(
        original_text=text,
        polygon_id=int(polygon_match.group(1)) if polygon_match else None,
        planning_typology=kb.resolve_planning_typology(text),
        explicit_lcz_type=explicit,
        preferences=preferences,
        constraints=constraints,
    )
