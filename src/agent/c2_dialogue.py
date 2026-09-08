"""Deterministic, auditable dialogue helpers for Layer Alterator C2."""

from __future__ import annotations

import re

from src.layer_alterator_agent.reference_loader import PREDICTOR_COLUMNS


COMPONENTS = {
    "trees": ("F_TV",),
    "tree": ("F_TV",),
    "树木": ("F_TV",),
    "乔木": ("F_TV",),
    "grass": ("F_G",),
    "low plants": ("F_G",),
    "草地": ("F_G",),
    "低矮植被": ("F_G",),
    "water": ("F_W",),
    "pond": ("F_W",),
    "水体": ("F_W",),
    "paved": ("F_AC", "IMD"),
    "impervious": ("F_AC", "IMD"),
    "硬化": ("F_AC", "IMD"),
    "不透水": ("F_AC", "IMD"),
}

SCOPE_GROUPS = {
    "vegetation": {"F_TV", "F_G", "TCH"},
    "植被": {"F_TV", "F_G", "TCH"},
    "building": {"BH", "BSF", "F_S", "F_M"},
    "建筑": {"BH", "BSF", "F_S", "F_M"},
    "water": {"F_W"},
    "水体": {"F_W"},
    "impervious": {"IMD", "F_AC"},
    "不透水": {"IMD", "F_AC"},
}


def empty_percentages() -> dict[str, float]:
    return {name: 0.0 for name in PREDICTOR_COLUMNS}


def extract_intensity(text: str) -> float | None:
    lowered = text.lower()
    labels = {
        "low": 25.0, "slight": 25.0, "轻度": 25.0,
        "moderate": 50.0, "medium": 50.0, "中度": 50.0,
        "strong": 75.0, "substantial": 75.0, "强": 75.0,
        "full": 100.0, "complete": 100.0, "完全": 100.0,
    }
    for label, value in labels.items():
        if label in lowered:
            return value
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if match:
        value = float(match.group(1))
        return value if 0 < value <= 100 else None
    return None


def extract_predictor_scope(text: str) -> list[str]:
    selected = set()
    lowered = text.lower()
    if any(phrase in lowered for phrase in [
        "all predictors", "all variables", "everything", "全部变量", "所有变量"
    ]):
        return list(PREDICTOR_COLUMNS)
    for predictor in PREDICTOR_COLUMNS:
        if re.search(rf"\b{re.escape(predictor.lower())}\b", lowered):
            selected.add(predictor)
    for phrase, predictors in SCOPE_GROUPS.items():
        if phrase in lowered:
            selected.update(predictors)
    for phrase, predictors in COMPONENTS.items():
        if phrase in lowered:
            selected.update(predictors)
    return [name for name in PREDICTOR_COLUMNS if name in selected]


def extract_polygon_id(text: str, polygon_ids) -> int | None:
    match = re.search(r"(?:polygon|zone|地块|区域)\s*#?\s*([A-Za-z0-9_.-]+)", text, re.I)
    if not match:
        return None
    # Do not treat sentence punctuation as part of a numeric/string zone ID.
    # For example, "Polygon 7." must resolve to the uploaded ID 7, not "7.".
    token = match.group(1).rstrip(".,;:!?")
    for polygon_id in polygon_ids:
        if str(polygon_id).lower() == token.lower():
            return polygon_id
    return None


def _direction_near(text: str, start: int) -> float:
    context = text[max(0, start - 18): start].lower()
    if re.search(r"decreas|reduc|less|remove|降低|减少|下降", context):
        return -1.0
    return 1.0


def _explicit_predictors(text: str) -> dict[str, float]:
    values = {}
    for predictor in PREDICTOR_COLUMNS:
        aliases = [predictor]
        if predictor == "SVF":
            aliases.append("SFV")
        alias_pattern = "|".join(map(re.escape, aliases))
        patterns = [
            rf"(?:{alias_pattern})\s*(?:=|:|by|增加|减少)?\s*([+-]?\d+(?:\.\d+)?)\s*%",
            rf"(?:increase|raise|decrease|reduce|增加|减少|降低)\s*(?:{alias_pattern})\s*(?:by)?\s*(\d+(?:\.\d+)?)\s*%",
            rf"(?:increase|raise|decrease|reduce|增加|减少|降低)\s*(?:{alias_pattern})"
            rf"\s*(?:in|for|于|在)?\s*(?:(?:polygon|zone|地块|区域)\s*#?\s*[A-Za-z0-9_.-]+)?"
            rf"\s*(?:by|增加|减少)?\s*(\d+(?:\.\d+)?)\s*%",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.I)
            if match:
                number = float(match.group(1))
                if not match.group(1).startswith(("+", "-")):
                    number *= _direction_near(text, match.start())
                values[predictor] = number
                break
    return values


def _all_predictor_change(text: str) -> dict[str, float]:
    """Parse one explicit percentage applied to all 12 predictors."""
    lowered = text.lower()
    if not any(phrase in lowered for phrase in (
        "all predictors", "all variables", "全部变量", "所有变量"
    )):
        return {}
    match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return {}
    number = float(match.group(1))
    if not match.group(1).startswith(("+", "-")):
        number *= _direction_near(text, match.start())
    return {name: number for name in PREDICTOR_COLUMNS}


def _component_changes(text: str) -> tuple[dict[str, float], list[str]]:
    changes, components_without_magnitude = {}, []
    lowered = text.lower()
    for phrase, predictors in COMPONENTS.items():
        for match in re.finditer(re.escape(phrase), lowered, re.I):
            after = text[match.end(): match.end() + 38]
            before = text[max(0, match.start() - 24): match.start()]
            number_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", after)
            context = after
            if not number_match:
                number_match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", before)
                context = before
            if not number_match:
                components_without_magnitude.extend(predictors)
                continue
            number = float(number_match.group(1))
            if not number_match.group(1).startswith(("+", "-")):
                number *= _direction_near(text, match.start())
            for predictor in predictors:
                changes[predictor] = number
    return changes, list(dict.fromkeys(components_without_magnitude))


def start_c2_proposal(text: str, polygon_ids) -> dict:
    polygon_id = extract_polygon_id(text, polygon_ids)
    if polygon_id is None:
        mentioned = re.search(
            r"(?:polygon|zone|地块|区域)\s*#?\s*([A-Za-z0-9_.-]+)",
            text,
            re.I,
        )
        available = ", ".join(str(value) for value in polygon_ids)
        if mentioned:
            invalid_id = mentioned.group(1).rstrip(".,;:!?")
            question = (
                f"Polygon {invalid_id} was not found in the uploaded vector. "
                f"Available Polygon IDs: {available}."
            )
        else:
            question = (
                "Which Polygon should this percentage adjustment apply to? "
                f"Available Polygon IDs: {available}."
            )
        return {
            "status": "needs_polygon",
            "question": question,
            "original_request": text,
        }
    changes = _all_predictor_change(text) or _explicit_predictors(text)
    component_changes, missing_magnitude = _component_changes(text)
    changes.update(component_changes)
    if not changes and not missing_magnitude:
        return {
            "status": "needs_component",
            "polygon_id": polygon_id,
            "question": (
                f"Which physical component should change in Polygon {polygon_id}: "
                "trees, grass, water, paved/impervious surface, or an exact predictor?"
            ),
            "original_request": text,
        }
    if missing_magnitude:
        return {
            "status": "needs_magnitude",
            "polygon_id": polygon_id,
            "components": missing_magnitude,
            "changes": changes,
            "question": (
                "What percentage change should I use? Give a signed value such as "
                "+30% or -10%."
            ),
            "original_request": text,
        }
    return complete_c2_proposal(polygon_id, changes, text)


def continue_c2_proposal(pending: dict, answer: str, polygon_ids) -> dict:
    combined = f"{pending.get('original_request', '')} {answer}".strip()
    if pending["status"] == "needs_polygon":
        answered_id = extract_polygon_id(answer, polygon_ids)
        if answered_id is not None:
            original = re.sub(
                r"(?:polygon|zone|地块|区域)\s*#?\s*[A-Za-z0-9_.-]+",
                f"Polygon {answered_id}",
                pending.get("original_request", ""),
                flags=re.I,
            )
            if not re.search(r"(?:polygon|zone|地块|区域)", original, re.I):
                original = f"{original} Polygon {answered_id}"
            return start_c2_proposal(original, polygon_ids)
        return start_c2_proposal(combined, polygon_ids)
    if pending["status"] == "needs_component":
        return start_c2_proposal(combined, polygon_ids)
    if pending["status"] == "needs_magnitude":
        match = re.search(r"([+-]?\d+(?:\.\d+)?)\s*%", answer)
        if not match:
            return {**pending, "question": "Please provide a percentage such as +30% or -10%."}
        number = float(match.group(1))
        if not match.group(1).startswith(("+", "-")):
            number *= _direction_near(answer, match.start())
        changes = dict(pending.get("changes", {}))
        for predictor in pending["components"]:
            changes[predictor] = number
        return complete_c2_proposal(pending["polygon_id"], changes, combined)
    return pending


def complete_c2_proposal(polygon_id, changes, original_request) -> dict:
    percentages = empty_percentages()
    percentages.update({key: float(value) for key, value in changes.items()})
    active_predictors = [name for name in PREDICTOR_COLUMNS if name in changes]
    return {
        "status": "ready",
        "polygon_id": polygon_id,
        "percentages": percentages,
        "active_predictors": active_predictors,
        "rule_mode": "C2" if set(active_predictors) == set(PREDICTOR_COLUMNS) else "C3",
        "original_request": original_request,
    }


def format_c2_proposal(proposal: dict) -> str:
    changed = [
        f"- `{name}`: **{value:+g}%**"
        for name, value in proposal["percentages"].items()
        if value != 0
    ]
    active = set(proposal.get("active_predictors", ()))
    inactive = [name for name in PREDICTOR_COLUMNS if name not in active]
    zero_pct = [name for name in active if proposal["percentages"][name] == 0]
    return "\n".join([
        f"### {proposal.get('rule_mode', 'C3')} direct percentage proposal for Polygon {proposal['polygon_id']}",
        "", "Percentage changes:", *(changed or ["- No non-zero change detected"]),
        *(["", f"Explicit 0% PCT predictors: {', '.join(zero_pct)}"] if zero_pct else []),
        "", f"NONE predictors: {', '.join(inactive) if inactive else 'none (all predictors use PCT)'}",
        "", "The seven fraction layers will be normalized per pixel after the change.",
        "Say `Confirm` to accept, `Revise` to discard, or provide corrected percentages.",
    ])


def format_grounded_transition(proposal: dict, rule_mode: str) -> str:
    changed = [
        name for name in PREDICTOR_COLUMNS
        if name in proposal["active_predictors"]
    ]
    lines = [
        f"### {rule_mode} transition proposal for Polygon {proposal['polygon_id']}",
        "",
        f"Target LCZ type: **{proposal['target_type']}**",
        f"Transition intensity: **{proposal['intensity']:g}%**",
        "",
        "| Predictor | Current mean | LCZ reference | Transition target | Input PCT | Estimated output |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in changed:
        lines.append(
            f"| {name} | {proposal['current_values'][name]:.3f} | "
            f"{proposal['target_values'][name]:.3f} | "
            f"{proposal['intermediate_targets'][name]:.3f} | "
            f"{proposal['percentages'][name]:+.1f}% | "
            f"{proposal['estimated_output_means'][name]:.3f} |"
        )
    if rule_mode == "C2":
        lines.extend(["", "All 12 rules will be `pct`; values shown as 0% remain PCT rules."])
    else:
        none_names = [name for name in PREDICTOR_COLUMNS if name not in changed]
        lines.extend([
            "",
            f"PCT predictors: {', '.join(changed)}",
            f"NONE predictors: {', '.join(none_names)}",
            "Fraction NONE layers receive no direct percentage, but joint fraction normalization may change them indirectly.",
        ])
    lines.extend([
        "",
        "Numerical source: current polygon raster means + professor LCZ reference + confirmed intensity.",
        "For fraction layers, Input PCT is the multiplicative Layer Alterator input. "
        "The Estimated output includes joint sum-to-one normalization, so its direction "
        "should be compared with Current mean.",
    ])
    if proposal.get("blockers"):
        lines.extend(["", "Cannot safely execute this relative transition:"])
        lines.extend(f"- {item}" for item in proposal["blockers"])
        lines.append("Use C1 replacement, revise the target, or keep those predictors unchanged.")
    else:
        lines.append("\nSay `Confirm` to accept or `Revise` to discard.")
    if proposal.get("warnings"):
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {item}" for item in proposal["warnings"])
    return "\n".join(lines)


def format_c2_status(confirmed: dict, unchanged: list, polygon_ids) -> str:
    lines = ["### Current percentage-adjustment plan", ""]
    for polygon_id in polygon_ids:
        if polygon_id in confirmed:
            changed = [f"{k} {v:+g}%" for k, v in confirmed[polygon_id].items() if v]
            lines.append(f"- Polygon {polygon_id}: {', '.join(changed) or 'all predictors 0%'}")
        elif polygon_id in unchanged:
            lines.append(f"- Polygon {polygon_id}: unchanged")
        else:
            lines.append(f"- Polygon {polygon_id}: not defined")
    return "\n".join(lines)


def merge_confirmed_percentage_proposal(confirmed, unchanged, polygon_ids, proposal):
    """Merge one confirmed polygon proposal without clearing earlier polygons."""
    merged = dict(confirmed)
    polygon_id = proposal["polygon_id"]
    merged[polygon_id] = dict(proposal["percentages"])
    remaining_unchanged = [pid for pid in unchanged if pid != polygon_id]
    active_predictors = [
        name
        for name in PREDICTOR_COLUMNS
        if any(values.get(name, 0) != 0 for values in merged.values())
    ]
    return merged, remaining_unchanged, active_predictors
