"""Pure helpers for the Streamlit planning workspace."""

import io
import json
import zipfile
from pathlib import Path


STATUS_COLORS = {
    "unplanned": "#94a3b8",
    "interviewing": "#f59e0b",
    "proposed": "#3b82f6",
    "confirmed": "#16a34a",
    "unchanged": "#64748b",
    "conflict": "#dc2626",
}


def polygon_rows(gdf, id_column, decisions, unchanged=(), pending_id=None):
    rows = []
    unchanged = set(unchanged)
    for _, feature in gdf.iterrows():
        polygon_id = feature[id_column]
        if polygon_id in unchanged:
            status, target = "unchanged", None
        elif polygon_id in decisions and str(decisions[polygon_id]).strip():
            status, target = "confirmed", decisions[polygon_id]
        elif polygon_id == pending_id:
            status, target = "proposed", None
        else:
            status, target = "unplanned", None
        rows.append({
            "Polygon": polygon_id,
            "Original type": feature.get("transf_type", ""),
            "Target LCZ type": target,
            "Status": status,
        })
    return rows


def generation_checklist(has_vector, goal, polygon_ids, decisions, unchanged=()):
    covered = set(decisions) | set(unchanged)
    checks = {
        "Vector loaded": bool(has_vector),
        "Goal defined": bool(str(goal).strip()),
        "Every polygon decided or unchanged": bool(polygon_ids)
        and set(polygon_ids) == covered,
        "At least one polygon will be transformed": bool(decisions),
    }
    return checks, all(checks.values())


def workflow_progress(has_vector, has_goal, polygon_ids, descriptions, unchanged, has_outputs):
    decided = {pid for pid, value in descriptions.items() if str(value).strip()}
    all_polygons = bool(polygon_ids) and (
        decided | set(unchanged) == set(polygon_ids)
    )
    vector_ready = bool(has_vector)
    goal_ready = vector_ready and bool(has_goal)
    polygons_ready = goal_ready and all_polygons
    outputs_ready = polygons_ready and bool(has_outputs)
    flags = [vector_ready, goal_ready, polygons_ready, outputs_ready]
    return flags, sum(flags) / len(flags)


def mark_unchanged(unchanged_polygon_ids, polygon_id):
    """Add once while preserving AgentState's list-based serialisable schema."""
    if polygon_id not in unchanged_polygon_ids:
        unchanged_polygon_ids.append(polygon_id)
    return unchanged_polygon_ids


def build_qgis_qml(field_name="target_lcz_type"):
    colors = {
        "Dense trees": "#1b7837", "Scattered trees": "#5aae61",
        "Low Plants": "#a6d96a", "Water": "#2b83ba",
        "Bare rock or paved": "#969696", "Bare Soil or sand": "#d8b365",
    }
    categories = "\n".join(
        f'<category value="{name}" label="{name}" symbol="{index}"/>'
        for index, name in enumerate(colors)
    )
    symbols = "\n".join(
        f'<symbol name="{index}" type="fill"><layer class="SimpleFill">'
        f'<Option name="color" value="{color}"/><Option name="outline_color" value="#333333"/>'
        f'</layer></symbol>' for index, color in enumerate(colors.values())
    )
    return f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3">
 <renderer-v2 type="categorizedSymbol" attr="{field_name}">
  <categories>{categories}</categories><symbols>{symbols}</symbols>
 </renderer-v2>
 <labeling type="simple"><settings><text-style fieldName="'Polygon ' || &quot;fid&quot;" isExpression="1"/></settings></labeling>
</qgis>"""


def add_delivery_bundle(result):
    """Create QGIS style and a portable ZIP next to generated files."""
    output_dir = Path(result["updated_vector_path"]).parent
    qml_path = output_dir / "urban_layer_agent.qml"
    qml_path.write_text(build_qgis_qml(), encoding="utf-8")
    zip_path = output_dir / "urban_layer_agent_results.zip"
    candidates = [
        result.get("updated_vector_path"), result.get("rules_path"),
        result.get("config_path"), result.get("explanation_path"),
        result.get("decision_evidence_path"), str(qml_path),
    ]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for candidate in candidates:
            if candidate and Path(candidate).exists():
                archive.write(candidate, Path(candidate).name)
    result["qml_path"] = str(qml_path)
    result["bundle_path"] = str(zip_path)
    return result
