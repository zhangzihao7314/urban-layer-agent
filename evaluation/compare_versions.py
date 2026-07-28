"""Create a transparent comparison of the three thesis prototype stages."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def compare():
    baseline = load("evaluation/baseline_report.json")
    table_agent = load("evaluation/agent_report.json")
    dialogue = load("evaluation/dialogue_report.json")
    elicitation = load("evaluation/elicitation_report.json")
    return {
        "warning": "The stages use different small development sets; values are engineering indicators, not a controlled final experiment.",
        "fixed_keyword_baseline": {
            "intent_accuracy": baseline.get("intent_accuracy"),
            "urban_type_accuracy": baseline.get("urban_type_accuracy"),
            "multi_turn_requirement_elicitation": False,
        },
        "table_driven_workflow": {
            "typology_accuracy": table_agent.get("planning_typology_accuracy"),
            "ranking_accuracy": table_agent.get("candidate_ranking_accuracy"),
            "multi_turn_requirement_elicitation": False,
        },
        "proposed_elicitation_agent": {
            "dialogue_success_rate": dialogue.get("dialogue_success_rate"),
            "requirement_task_success_rate": elicitation.get("task_success_rate"),
            "average_clarification_turns": elicitation.get("average_clarification_turns"),
            "repeated_question_rate": elicitation.get("repeated_question_rate"),
            "multi_turn_requirement_elicitation": True,
        },
    }


if __name__ == "__main__":
    result = compare()
    (ROOT / "evaluation/version_comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
