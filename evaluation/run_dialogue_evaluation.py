"""Evaluate multi-turn planning context without requiring an LLM service."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.conversation_controller import ConversationController
from src.agent.schemas import AgentState


def evaluate(path=ROOT / "evaluation/dialogue_cases.json"):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    details = []
    for case in cases:
        state = AgentState(
            polygon_ids=[7, 8, 9],
            confirmed_decisions={int(k): v for k, v in case.get("initial_decisions", {}).items()},
        )
        result = None
        for message in case["messages"]:
            result = ConversationController().interpret(message, state)
        passed = True
        expected = case.get("expected", {})
        for key, urban_type in expected.items():
            if key == "unchanged":
                passed &= result.keep_unchanged == urban_type
            else:
                proposal = result.proposals.get(int(key))
                passed &= bool(proposal and proposal.recommended_type == urban_type)
        excluded = case.get("expected_excluded_type")
        if excluded:
            passed &= all(excluded not in proposal.candidate_types for proposal in result.proposals.values())
        warning = case.get("expected_warning_contains")
        if warning:
            passed &= any(warning in item for item in result.warnings)
        details.append({"name": case["name"], "passed": bool(passed)})
    return {
        "case_count": len(details),
        "passed": sum(item["passed"] for item in details),
        "dialogue_success_rate": sum(item["passed"] for item in details) / len(details),
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate()
    (ROOT / "evaluation/dialogue_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
