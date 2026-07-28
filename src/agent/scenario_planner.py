"""Compare alternative polygon transformation scenarios."""

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.agent.goal_reasoner import GoalRecommendation


@dataclass(frozen=True)
class ScenarioAssessment:
    name: str
    decisions: Dict[int, str]
    score: float
    warnings: List[str]
    explanation: str


def compare_scenarios(
    scenarios: Dict[str, Dict[int, str]],
    recommendation: GoalRecommendation,
) -> List[ScenarioAssessment]:
    results = []
    for name, decisions in scenarios.items():
        values = [recommendation.scores.get(target, 0.0) for target in decisions.values()]
        score = round(sum(values) / len(values), 4) if values else 0.0
        warnings = []
        if decisions and len(set(decisions.values())) == 1:
            warnings.append("All polygons use one LCZ type; check whether spatial diversity is realistic.")
        if "Water" in decisions.values():
            warnings.append("Water conversion requires a feasibility check; LCZ suitability alone is insufficient.")
        results.append(ScenarioAssessment(
            name, decisions, score, warnings,
            f"Mean suitability for {recommendation.goal}: {score:.4f}.",
        ))
    return sorted(results, key=lambda item: item.score, reverse=True)
