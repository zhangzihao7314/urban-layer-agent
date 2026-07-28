"""Evaluate requirement completion, question relevance, and clarification turns."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.elicitation import RequirementElicitationLoop, check_completeness
from src.agent.proposal_engine import score_requirement


def evaluate(path=ROOT / "evaluation/elicitation_cases.json"):
    cases = json.loads(Path(path).read_text(encoding="utf-8"))
    details, total_questions, repeats = [], 0, 0
    for case in cases:
        loop = RequirementElicitationLoop(max_questions=case["max_questions"])
        requirement = loop.start(7, case["initial"], "urban planning")
        questions = []
        question = loop.next_question(requirement)
        if question:
            questions.append(question)
        for answer in case["answers"]:
            loop.answer(requirement, answer)
            question = loop.next_question(requirement)
            if question:
                questions.append(question)
        complete = check_completeness(requirement).complete
        candidates = score_requirement(requirement) if complete else []
        top = candidates[0].urban_type if candidates else None
        passed = complete
        if case.get("expected_type"):
            passed &= top == case["expected_type"]
        if case.get("excluded_type"):
            passed &= all(item.urban_type != case["excluded_type"] or item.conflict_penalty > 0 for item in candidates)
        repeated = len(questions) - len(set(questions))
        repeats += repeated
        total_questions += len(questions)
        details.append({
            "name": case["name"], "complete": complete, "top_type": top,
            "question_count": len(questions), "repeated_questions": repeated,
            "passed": bool(passed),
        })
    return {
        "case_count": len(details),
        "task_success_rate": sum(item["passed"] for item in details) / len(details),
        "average_clarification_turns": total_questions / len(details),
        "repeated_question_rate": repeats / total_questions if total_questions else 0,
        "details": details,
    }


if __name__ == "__main__":
    report = evaluate()
    (ROOT / "evaluation/elicitation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
