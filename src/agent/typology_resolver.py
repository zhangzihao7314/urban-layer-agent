from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List

from src.agent.schemas import TypologyProposal
from src.agent.context_parser import parse_planning_context
from src.agent.knowledge_base import get_knowledge_base
from src.layer_alterator_agent.matcher import SUPPORTED_URBAN_TYPES, URBAN_TYPE_ALIASES


def propose_typology(
    polygon_id: int,
    description: str,
    constraints: list[str] | None = None,
) -> TypologyProposal:
    text = description.lower().strip()
    kb = get_knowledge_base()
    context = parse_planning_context(description, kb)
    if context.planning_typology and not context.explicit_lcz_type:
        reference_candidates = kb.candidates_for(context.planning_typology, include_occasional=False)
        excluded_types = {
            constraint.split(":", 1)[1]
            for constraint in (constraints or [])
            if constraint.startswith("exclude:")
        }
        reference_candidates = [
            item for item in reference_candidates if item.urban_type not in excluded_types
        ]
        preference_boosts = {
            "trees": {"Dense trees": 2.0, "Scattered trees": 0.75},
            "grass": {"Low Plants": 2.0},
            "water": {"Water": 2.0},
            "open": {"Open midrise": 1.0, "Open low-rise": 1.0},
            "paved": {"Bare rock or paved": 2.0},
        }
        reference_candidates = sorted(
            reference_candidates,
            key=lambda item: -(
                item.score + sum(
                    preference_boosts.get(preference, {}).get(item.urban_type, 0.0)
                    for preference in context.preferences
                )
            ),
        )
        choices = [item.urban_type for item in reference_candidates]
        if choices:
            # A specific preference such as "grass park" or "wooded park"
            # is sufficient to form a proposal; a generic "park" still
            # requires clarification.
            preference_is_decisive = bool(context.preferences)
            recommended = choices[0] if preference_is_decisive else None
            return TypologyProposal(
                polygon_id=polygon_id,
                original_description=description,
                candidate_types=choices,
                recommended_type=recommended,
                confidence=0.80 if preference_is_decisive else 0.55,
                needs_clarification=not preference_is_decisive,
                clarification_question=(
                    f"{context.planning_typology} has several valid LCZ matches in the reference table. "
                    f"Which type best describes Polygon {polygon_id}: {', '.join(choices)}?"
                ) if not preference_is_decisive else None,
                planning_typology=context.planning_typology,
                candidate_evidence=[item.evidence for item in reference_candidates],
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
        planning_typology=context.planning_typology,
    )
