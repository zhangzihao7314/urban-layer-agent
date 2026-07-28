"""Build auditable decision records and human-readable evidence reports."""

from datetime import datetime, timezone
from pathlib import Path
import json


def build_decision_record(polygon_id, original_request, planning_typology, candidate_evidence,
                          final_type, predictor_values, confirmed=True):
    return {
        "polygon_id": polygon_id,
        "original_request": original_request,
        "planning_typology": planning_typology,
        "candidate_evidence": candidate_evidence,
        "final_lcz_type": final_type,
        "predictor_values": predictor_values,
        "confirmed_by_user": confirmed,
    }


def write_evidence_bundle(output_dir, goal, records, source_files):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "goal": goal,
        "reference_sources": [str(Path(path)) for path in source_files],
        "decisions": records,
    }
    json_path = output_dir / "decision_evidence.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Simulation decision evidence", "", f"Goal: {goal}", ""]
    for record in records:
        lines.extend([
            f"## Polygon {record['polygon_id']}",
            f"- Original request: {record['original_request']}",
            f"- Planning typology: {record['planning_typology'] or 'explicit LCZ request'}",
            f"- Final LCZ type: {record['final_lcz_type']}",
            f"- User confirmed: {record['confirmed_by_user']}",
            f"- Evidence: {'; '.join(record['candidate_evidence']) or 'explicit target type'}",
            "",
        ])
    md_path = output_dir / "simulation_explanation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"decision_evidence_path": str(json_path), "explanation_path": str(md_path)}
