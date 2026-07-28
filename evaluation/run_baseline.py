import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.intent_router import route_intent
from src.layer_alterator_agent.matcher import match_urban_type


def evaluate(cases_path: Path) -> dict:
    rows = list(csv.DictReader(cases_path.open(encoding="utf-8")))
    intent_correct = 0
    type_correct = 0
    type_total = 0
    details = []
    for row in rows:
        decision = route_intent(row["text"])
        intent_ok = decision.intent.value == row["expected_intent"]
        intent_correct += int(intent_ok)
        predicted_type = ""
        type_ok = None
        if row["expected_type"]:
            type_total += 1
            description = decision.target_description or row["text"]
            try:
                predicted_type = match_urban_type(description)
            except ValueError:
                predicted_type = ""
            type_ok = predicted_type == row["expected_type"]
            type_correct += int(type_ok)
        details.append({**row, "predicted_intent": decision.intent.value, "predicted_type": predicted_type, "intent_ok": intent_ok, "type_ok": type_ok})
    return {
        "case_count": len(rows),
        "intent_accuracy": intent_correct / len(rows),
        "urban_type_accuracy": type_correct / type_total if type_total else None,
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate(PROJECT_ROOT / "evaluation/cases.csv")
    output = PROJECT_ROOT / "evaluation/baseline_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
