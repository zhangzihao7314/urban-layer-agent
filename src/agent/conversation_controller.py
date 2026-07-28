"""Controller that converts flexible dialogue into safe, executable plans."""

from dataclasses import dataclass, field
from typing import Dict, List

from src.agent.flexible_dialogue import (
    DialogueInterpretation,
    detect_conflicts,
    parse_flexible_message,
)
from src.agent.schemas import AgentState, TypologyProposal
from src.agent.typology_resolver import propose_typology


@dataclass
class ControllerResult:
    kind: str
    interpretation: DialogueInterpretation
    proposals: Dict[int, TypologyProposal] = field(default_factory=dict)
    keep_unchanged: List[int] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)


class ConversationController:
    def interpret(self, text: str, state: AgentState) -> ControllerResult:
        parsed = parse_flexible_message(text)
        warnings = detect_conflicts(parsed.instructions, parsed.constraints)
        if parsed.command:
            return ControllerResult(parsed.command, parsed, warnings=warnings)
        if len(parsed.instructions) < 2 and not any(
            item.action == "copy_previous" for item in parsed.instructions
        ):
            return ControllerResult("single_or_other", parsed, warnings=warnings)

        proposals = {}
        keep = []
        previous_type = next(reversed(state.confirmed_decisions.values()), None) if state.confirmed_decisions else None
        for instruction in parsed.instructions:
            if instruction.polygon_id not in state.polygon_ids:
                warnings.append(f"Unknown polygon ID: {instruction.polygon_id}.")
                continue
            if instruction.action == "keep":
                keep.append(instruction.polygon_id)
                continue
            description = instruction.description
            if instruction.action == "copy_previous":
                if not previous_type:
                    warnings.append(f"Polygon {instruction.polygon_id}: there is no previous choice to copy.")
                    continue
                description = previous_type
            proposals[instruction.polygon_id] = propose_typology(
                instruction.polygon_id, description, state.constraints + parsed.constraints
            )
        return ControllerResult(
            "multi_plan",
            parsed,
            proposals,
            keep,
            warnings,
            ["parse_flexible_message", "search_planning_typology", "rank_lcz_candidates"],
        )


def explain_grounded_choice(proposal: TypologyProposal, question: str = "") -> str:
    evidence = proposal.candidate_evidence or ["The request explicitly named an LCZ type."]
    selected = proposal.recommended_type or (
        proposal.candidate_types[0] if proposal.candidate_types else "not selected"
    )
    alternatives = [item for item in proposal.candidate_types if item != selected]
    lines = [
        f"Current leading option: **{selected}**.",
        "",
        "**Reference-table facts**",
        *[f"- {item}" for item in evidence],
        "",
        "**Agent reasoning**",
        f"- The ordering also reflects the wording of your request: {proposal.original_description}.",
    ]
    if alternatives:
        lines.append(f"- Alternatives still available: {', '.join(alternatives)}.")
    lines.extend([
        "",
        "**Limitation**",
        "- LCZ correspondence and predictor values do not prove real-world construction feasibility.",
    ])
    return "\n".join(lines)
