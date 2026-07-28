"""Rule classifier for notebook cases C0-C5."""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RuleClassification:
    code: str
    executable: bool
    description: str


def classify_rules(rules: Dict[str, str]) -> RuleClassification:
    values = {
        str(value.get("action", "none") if isinstance(value, dict) else value).lower()
        for value in rules.values()
    }
    if not values or values == {"none"}:
        return RuleClassification("C0", True, "No transformation")
    if values == {"mask"}:
        return RuleClassification("C1", True, "Polygon masking")
    if values == {"pct"}:
        return RuleClassification("C2", True, "Percentage transformation")
    if values <= {"pct", "none"}:
        return RuleClassification("C3", True, "Partial percentage transformation")
    if values <= {"mask", "none"}:
        return RuleClassification("C4", False, "Mask and none cannot be mixed")
    if "mask" in values and "pct" in values:
        return RuleClassification("C5", False, "Mask and percentage cannot be mixed")
    raise ValueError(f"Unknown Layer Alterator rule values: {sorted(values)}")


def require_executable(rules: Dict[str, str]) -> RuleClassification:
    result = classify_rules(rules)
    if not result.executable:
        raise ValueError(f"{result.code}: {result.description}")
    return result
