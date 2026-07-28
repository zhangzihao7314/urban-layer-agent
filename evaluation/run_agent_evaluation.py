"""Evaluate table-backed typology interpretation and ranking."""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.tools import UrbanLayerTools


def evaluate(cases_path=ROOT / "evaluation/agent_cases.csv"):
    tools = UrbanLayerTools(
        ROOT / "data/reference/LCZ_urban_types_match.xlsx",
        ROOT / "data/reference/ref_predictor_values_mean.csv",
    )
    cases = list(csv.DictReader(Path(cases_path).open(encoding="utf-8")))
    details = []
    for case in cases:
        search = tools.search_planning_typology(case["text"])
        ranked = tools.rank_lcz_candidates(case["text"])
        predicted_typology = search["context"]["planning_typology"] or ""
        predicted_top = ranked[0]["urban_type"] if ranked else ""
        accepted = case["expected_top_group"].split("|")
        details.append({
            **case,
            "predicted_planning_typology": predicted_typology,
            "predicted_top_type": predicted_top,
            "typology_ok": predicted_typology == case["expected_planning_typology"],
            "ranking_ok": predicted_top in accepted,
        })
    report = {
        "case_count": len(details),
        "planning_typology_accuracy": sum(row["typology_ok"] for row in details) / len(details),
        "candidate_ranking_accuracy": sum(row["ranking_ok"] for row in details) / len(details),
        "details": details,
    }
    return report


if __name__ == "__main__":
    report = evaluate()
    path = ROOT / "evaluation/agent_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "details"}, indent=2))
