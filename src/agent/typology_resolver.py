from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List

from src.agent.schemas import TypologyProposal
from src.layer_alterator_agent.matcher import SUPPORTED_URBAN_TYPES, URBAN_TYPE_ALIASES


AMBIGUOUS_CHOICES = {
    "park": ["Dense trees", "Scattered trees", "Low Plants"],
    "green area": ["Dense trees", "Scattered trees", "Low Plants"],
    "residential": ["Compact midrise", "Compact low-rise", "Open midrise", "Open low-rise"],
}


def propose_typology(polygon_id: int, description: str) -> TypologyProposal:
    text = description.lower().strip()
    for phrase, choices in AMBIGUOUS_CHOICES.items():
        if phrase in text and not any(choice.lower() in text for choice in choices):
            return TypologyProposal(
                polygon_id=polygon_id,
                original_description=description,
                candidate_types=choices,
                recommended_type=None,
                confidence=0.45,
                needs_clarification=True,
                clarification_question=f"Which type best describes Polygon {polygon_id}: {', '.join(choices)}?",
            )

    scored = []
    for alias, urban_type in URBAN_TYPE_ALIASES.items():
        score = 1.0 if alias in text else SequenceMatcher(None, text, alias).ratio()
        scored.append((score, urban_type))
    best_by_type = {}
    for score, urban_type in scored:
        best_by_type[urban_type] = max(score, best_by_type.get(urban_type, 0.0))
    ranked = sorted(best_by_type.items(), key=lambda item: item[1], reverse=True)
    candidates = [name for name, score in ranked[:3] if score >= 0.45]
    confidence = ranked[0][1] if ranked else 0.0
    recommended = ranked[0][0] if confidence >= 0.65 else None
    needs_clarification = recommended is None
    return TypologyProposal(
        polygon_id=polygon_id,
        original_description=description,
        candidate_types=candidates or SUPPORTED_URBAN_TYPES[:3],
        recommended_type=recommended,
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_question=(
            f"Please choose the closest target type for Polygon {polygon_id}: "
            f"{', '.join(candidates or SUPPORTED_URBAN_TYPES[:3])}."
            if needs_clarification else None
        ),
    )

