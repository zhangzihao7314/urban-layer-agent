"""Flexible multi-turn language utilities for the urban planning Agent."""

from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PolygonInstruction:
    polygon_id: int
    action: str
    description: str = ""


@dataclass
class DialogueInterpretation:
    instructions: List[PolygonInstruction] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    command: Optional[str] = None
    explanation_question: Optional[str] = None


def parse_flexible_message(text: str) -> DialogueInterpretation:
    normalized = text.lower().strip()
    interpretation = DialogueInterpretation()
    if any(phrase in normalized for phrase in ["undo", "go back", "撤销", "返回上一步"]):
        interpretation.command = "undo"
    elif any(phrase in normalized for phrase in ["what have i decided", "current plan", "summary", "目前决定", "当前方案"]):
        interpretation.command = "status"
    elif normalized in {"confirm all", "accept the plan", "looks good", "都确认", "确认全部"}:
        interpretation.command = "confirm_all"
    elif any(phrase in normalized for phrase in ["why", "why not", "where did", "which option", "为什么", "数据来源"]):
        interpretation.command = "explain"
        interpretation.explanation_question = text

    constraint_patterns = {
        "exclude:Water": ["do not use water", "without water", "no water", "不要水体"],
        "preserve_unchanged": ["leave unchanged", "keep unchanged", "do not change", "保持不变"],
        "pedestrian_access": ["walk", "pedestrian", "accessible", "步行", "行人"],
        "preserve_existing_trees": ["existing trees", "already exist", "现有树木"],
    }
    interpretation.constraints = [
        constraint for constraint, phrases in constraint_patterns.items()
        if any(phrase in normalized for phrase in phrases)
    ]

    same_match = re.search(
        r"(?:same choice|same type|apply (?:that|it|the same choice))\s+(?:to\s+)?(?:polygon|zone)\s*(\d+)",
        normalized,
    )
    if same_match:
        interpretation.instructions.append(
            PolygonInstruction(int(same_match.group(1)), "copy_previous")
        )
        return interpretation

    matches = list(re.finditer(r"(?:polygon|zone|区域)\s*(\d+)", text, re.IGNORECASE))
    for index, match in enumerate(matches):
        polygon_id = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segment = text[match.end():end].strip(" ,;:.，。")
        segment = re.sub(
            r"^(?:should\s+become|change\s+to|become|set\s+to|into|as|should\s+be|make)\s+",
            "", segment, flags=re.IGNORECASE,
        )
        segment = re.sub(
            r"\s*(?:,?\s+and\s+)?(?:leave|keep|do not change)\s*$",
            "",
            segment,
            flags=re.IGNORECASE,
        )
        prefix = text[max(0, match.start() - 25):match.start()]
        if (
            re.search(r"\b(?:leave|keep|remain)\b.*\bunchanged\b|\bdo not change\b", segment, re.IGNORECASE)
            or re.search(r"\b(?:leave|keep|do not change)\s*$", prefix, re.IGNORECASE)
            or segment.lower() in {"unchanged", "as it is", "the same"}
        ):
            action, description = "keep", ""
        else:
            action, description = "change", segment.strip(" ,;:.，。")
        interpretation.instructions.append(PolygonInstruction(polygon_id, action, description))

    return interpretation


def resolve_natural_selection(
    text: str,
    candidates: List[str],
    scores: Optional[Dict[str, float]] = None,
) -> Optional[str]:
    normalized = text.lower().strip()
    explicit = next((candidate for candidate in candidates if candidate.lower() in normalized), None)
    if explicit:
        return explicit
    ordinal = {
        "first": 0, "1": 0, "second": 1, "2": 1, "third": 2, "3": 2,
        "第一个": 0, "第二个": 1, "第三个": 2,
    }
    for phrase, index in ordinal.items():
        if phrase in normalized and index < len(candidates):
            return candidates[index]
    if any(phrase in normalized for phrase in ["greener", "most vegetation", "最绿", "植被最多"]):
        if scores:
            return max(candidates, key=lambda name: scores.get(name, float("-inf")))
        preferred = ["Dense trees", "Scattered trees", "Low Plants"]
        return next((name for name in preferred if name in candidates), None)
    return None


def format_plan_status(
    goal: str,
    decisions: Dict[int, str],
    constraints: List[str],
    unchanged_polygon_ids: Optional[List[int]] = None,
) -> str:
    lines = [f"Goal: {goal or 'not defined'}", "", "Current polygon decisions:"]
    lines.extend(
        [f"- Polygon {polygon_id}: {urban_type}" for polygon_id, urban_type in sorted(decisions.items())]
        or ["- No confirmed decisions yet."]
    )
    lines.extend(
        f"- Polygon {polygon_id}: unchanged"
        for polygon_id in sorted(unchanged_polygon_ids or [])
    )
    if constraints:
        lines.extend(["", "Active constraints:"] + [f"- {item}" for item in constraints])
    return "\n".join(lines)


def detect_conflicts(instructions: List[PolygonInstruction], constraints: List[str]) -> List[str]:
    warnings = []
    if "exclude:Water" in constraints:
        for item in instructions:
            description = item.description.lower()
            if "water" in description and not any(
                phrase in description for phrase in ["without water", "no water", "do not use water"]
            ):
                warnings.append(f"Polygon {item.polygon_id} requests Water but the plan excludes Water.")
    if "pedestrian_access" in constraints:
        warnings.append(
            "Pedestrian accessibility cannot be verified from the two LCZ reference tables; "
            "a path or pedestrian-network layer is required."
        )
    return warnings


def is_natural_confirmation(text: str) -> bool:
    normalized = text.lower().strip(" .!")
    return normalized in {
        "confirm", "yes", "accept", "looks good", "that is fine", "that's fine",
        "the first option is fine", "use it", "go ahead", "确认", "可以", "就这样",
    }
