"""Convert complete requirements into scored, auditable planning proposals."""

from dataclasses import asdict, dataclass, field
from typing import Dict, List

from src.agent.goal_reasoner import recommend_for_goal
from src.agent.knowledge_base import get_knowledge_base
from src.agent.requirements import PlanningRequirement


@dataclass(frozen=True)
class ScoredCandidate:
    urban_type: str
    lcz_code: int
    reference_score: float
    preference_score: float
    goal_score: float
    conflict_penalty: float
    total_score: float
    evidence: List[str]


@dataclass
class PlanningScenario:
    name: str
    candidate: ScoredCandidate
    strengths: List[str]
    limitations: List[str]
    predictor_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class CompleteProposal:
    polygon_id: int
    requirement_summary: Dict
    scenarios: List[PlanningScenario]
    recommended_scenario: str
    confidence: float
    needs_confirmation: bool = True
    external_data_needed: List[str] = field(default_factory=list)


PREFERENCE_SCORES = {
    "dense_trees": {"Dense trees": 2.5, "Scattered trees": 0.8},
    "scattered_trees": {"Scattered trees": 2.5, "Dense trees": 0.5, "Low Plants": 0.3},
    "low_plants": {"Low Plants": 2.5, "Scattered trees": 0.5},
}


def score_requirement(requirement: PlanningRequirement) -> List[ScoredCandidate]:
    kb = get_knowledge_base()
    if not requirement.planning_typology:
        raise ValueError("A planning typology is required before scoring.")
    reference = kb.candidates_for(requirement.planning_typology)
    goal = recommend_for_goal(requirement.overall_goal or requirement.primary_function or "")
    scored = []
    for item in reference:
        preference = PREFERENCE_SCORES.get(
            requirement.vegetation_preference or "", {}
        ).get(item.urban_type, 0.0)
        if requirement.openness_preference == "open":
            preference += {
                "Low Plants": 1.0, "Scattered trees": 0.8,
                "Open low-rise": 0.8, "Open midrise": 0.6,
            }.get(item.urban_type, 0.0)
        goal_score = max(0.0, goal.scores.get(item.urban_type, 0.0)) * 4
        penalty = 10.0 if f"exclude:{item.urban_type}" in requirement.constraints else 0.0
        total = round(item.score + preference + goal_score - penalty, 4)
        scored.append(ScoredCandidate(
            item.urban_type, item.lcz_code, item.score, round(preference, 3),
            round(goal_score, 3), penalty, total,
            [item.evidence, f"Preference contribution: {preference:+.3f}",
             f"Goal contribution: {goal_score:+.3f}", f"Constraint penalty: {-penalty:+.3f}"],
        ))
    return sorted(scored, key=lambda candidate: (-candidate.total_score, candidate.lcz_code))


def build_complete_proposal(requirement: PlanningRequirement, predictor_lookup) -> CompleteProposal:
    candidates = score_requirement(requirement)
    candidates = [item for item in candidates if item.conflict_penalty < 10]
    if not candidates:
        raise ValueError("No LCZ candidate satisfies the current constraints.")
    selected = candidates[:3]
    labels = ["Recommended", "Alternative A", "Alternative B"]
    scenarios = []
    for label, candidate in zip(labels, selected):
        strengths = [
            f"Reference correspondence score: {candidate.reference_score:.1f}",
            f"Requirement preference score: {candidate.preference_score:.1f}",
        ]
        limitations = ["LCZ suitability does not prove construction or legal feasibility."]
        scenarios.append(PlanningScenario(
            label, candidate, strengths, limitations,
            predictor_lookup(candidate.urban_type),
        ))
    gap = selected[0].total_score - selected[1].total_score if len(selected) > 1 else selected[0].total_score
    confidence = round(min(0.95, 0.55 + max(0, gap) * 0.1), 2)
    external = []
    if "pedestrian_access" in requirement.constraints:
        external.append("Pedestrian/path network")
    if requirement.primary_function == "cooling":
        external.append("Meteorological or LST model output for actual cooling impact")
    return CompleteProposal(
        requirement.polygon_id,
        requirement.to_dict(),
        scenarios,
        scenarios[0].name,
        confidence,
        True,
        external,
    )


def format_complete_proposal(proposal: CompleteProposal) -> str:
    lines = [
        "## My understanding",
        f"- Polygon: {proposal.polygon_id}",
        f"- Planning typology: {proposal.requirement_summary.get('planning_typology')}",
        f"- Main function: {proposal.requirement_summary.get('primary_function')}",
        f"- Vegetation: {proposal.requirement_summary.get('vegetation_preference')}",
        "",
        "## Scenarios",
    ]
    for scenario in proposal.scenarios:
        lines.extend([
            f"### {scenario.name}: {scenario.candidate.urban_type}",
            f"- Total score: {scenario.candidate.total_score:.3f}",
            f"- LCZ: {scenario.candidate.lcz_code}",
            f"- Strengths: {'; '.join(scenario.strengths)}",
            f"- Limitation: {'; '.join(scenario.limitations)}",
        ])
    lines.extend(["", f"Confidence: {proposal.confidence:.2f}"])
    if proposal.external_data_needed:
        lines.extend(["", "Additional data needed for full validation:"] +
                     [f"- {item}" for item in proposal.external_data_needed])
    lines.append("\nPlease confirm the recommended scenario or choose an alternative.")
    return "\n".join(lines)


def explain_scenario_comparison(proposal: CompleteProposal, question: str) -> str:
    normalized = question.lower()
    recommended = proposal.scenarios[0]
    target = next(
        (
            scenario for scenario in proposal.scenarios
            if scenario.candidate.urban_type.lower() in normalized
        ),
        None,
    )
    if target is None or target is recommended:
        return (
            f"**{recommended.candidate.urban_type}** is currently recommended with "
            f"a score of {recommended.candidate.total_score:.3f}.\n\n"
            + "\n".join(f"- {item}" for item in recommended.candidate.evidence)
        )
    difference = recommended.candidate.total_score - target.candidate.total_score
    return "\n".join([
        f"**Why {recommended.candidate.urban_type} instead of {target.candidate.urban_type}?**",
        "",
        f"- {recommended.candidate.urban_type}: {recommended.candidate.total_score:.3f}",
        f"- {target.candidate.urban_type}: {target.candidate.total_score:.3f}",
        f"- Score difference: {difference:.3f}",
        "",
        f"{recommended.candidate.urban_type} better matches the captured combination "
        "of reference correspondence, user preferences, goal, and constraints.",
        "",
        f"{target.candidate.urban_type} remains a valid alternative; choosing it is "
        "a planning preference, not a numerical error.",
    ])
