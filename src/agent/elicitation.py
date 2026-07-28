"""Requirement completeness checks and one-question-at-a-time elicitation."""

from dataclasses import dataclass
from typing import Callable, List, Optional

from src.agent.requirements import PlanningRequirement, update_requirement


@dataclass(frozen=True)
class CompletenessResult:
    complete: bool
    missing_fields: List[str]
    conflicts: List[str]
    next_question_topic: Optional[str]
    confidence: float


def check_completeness(requirement: PlanningRequirement) -> CompletenessResult:
    missing = []
    conflicts = []
    if not requirement.planning_typology:
        missing.append("planning_typology")
    if requirement.planning_typology == "Urban Park":
        if not requirement.primary_function:
            missing.append("primary_function")
        if not requirement.vegetation_preference:
            missing.append("vegetation_preference")
        if requirement.primary_function == "recreation" and not requirement.openness_preference:
            missing.append("openness_preference")
        if requirement.primary_function == "stormwater" and requirement.water_allowed is None:
            missing.append("water_allowed")
    if requirement.water_allowed is False and requirement.primary_function == "stormwater":
        conflicts.append("Stormwater management was requested while Water was excluded.")
    confidence = max(0.0, 1.0 - 0.2 * len(missing) - 0.25 * len(conflicts))
    return CompletenessResult(
        complete=not missing and not conflicts,
        missing_fields=missing,
        conflicts=conflicts,
        next_question_topic=missing[0] if missing else None,
        confidence=round(confidence, 2),
    )


QUESTIONS = {
    "planning_typology": "What kind of place should this polygon become—for example a park, urban forest, residential area, or waterfront?",
    "primary_function": "What is the main purpose: cooling, recreation, biodiversity, or stormwater management?",
    "vegetation_preference": "Should it be mainly dense trees, scattered trees with some shade, or open grass/low plants?",
    "openness_preference": "Should the space remain open for activities, or can it have a denser tree canopy?",
    "water_allowed": "May the proposal include a pond or other water surface?",
}


class RequirementElicitationLoop:
    def __init__(
        self,
        max_questions: int = 5,
        question_generator: Optional[Callable[[PlanningRequirement, str], str]] = None,
    ):
        self.max_questions = max_questions
        self.question_generator = question_generator

    def start(self, polygon_id: int, text: str, overall_goal: str = "") -> PlanningRequirement:
        requirement = PlanningRequirement(polygon_id, text, overall_goal=overall_goal or None)
        return update_requirement(requirement, text)

    def answer(self, requirement: PlanningRequirement, text: str) -> PlanningRequirement:
        return update_requirement(requirement, text)

    def next_question(self, requirement: PlanningRequirement) -> Optional[str]:
        result = check_completeness(requirement)
        if result.complete:
            return None
        if requirement.question_count >= self.max_questions:
            self._apply_defaults(requirement, result.missing_fields)
            return None
        requirement.question_count += 1
        question = QUESTIONS[result.next_question_topic]
        if self.question_generator:
            try:
                generated = self.question_generator(requirement, result.next_question_topic)
                if generated and 3 <= len(generated.split()) <= 45:
                    question = generated.strip()
            except Exception:
                pass
        if question in requirement.asked_questions:
            self._apply_defaults(requirement, [result.next_question_topic])
            return None
        requirement.asked_questions.append(question)
        return question

    @staticmethod
    def _apply_defaults(requirement: PlanningRequirement, missing: List[str]) -> None:
        defaults = {
            "primary_function": "cooling",
            "vegetation_preference": "scattered_trees",
            "openness_preference": "open",
            "water_allowed": False,
        }
        for field_name in missing:
            if field_name in defaults:
                setattr(requirement, field_name, defaults[field_name])
                requirement.assumptions.append(
                    f"{field_name} defaulted to {defaults[field_name]} after the clarification limit."
                )
