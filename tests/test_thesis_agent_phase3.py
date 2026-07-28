import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from src.agent.elicitation import RequirementElicitationLoop, check_completeness
from src.agent.execution_manager import (
    build_execution_plan,
    recommend_replanning,
    validate_raster_outputs,
)
from src.agent.proposal_engine import build_complete_proposal, score_requirement
from src.agent.proposal_engine import explain_scenario_comparison
from src.agent.requirements import PlanningRequirement, update_requirement
from src.layer_alterator_agent.reference_loader import (
    PREDICTOR_COLUMNS,
    get_predictor_values,
    load_reference_table,
)
from src.layer_alterator_agent.vector_writer import write_attributes_to_vector
from src.agent.failure_guard import require_reference_files, safe_domain_action, safe_llm_json


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "data/reference/ref_predictor_values_mean.csv"


def test_requirement_elicitation_repeats_until_complete():
    loop = RequirementElicitationLoop()
    requirement = loop.start(7, "a pleasant park", "reduce urban heat")
    assert loop.next_question(requirement).startswith("What is the main purpose")
    loop.answer(requirement, "Cooling and recreation are important.")
    assert "dense trees" in loop.next_question(requirement)
    loop.answer(requirement, "Scattered trees with some shade, and keep it open.")
    assert check_completeness(requirement).complete
    assert loop.next_question(requirement) is None


def test_elicitation_limit_records_assumptions():
    loop = RequirementElicitationLoop(max_questions=0)
    requirement = loop.start(7, "a park")
    assert loop.next_question(requirement) is None
    assert requirement.assumptions


def test_conflicting_stormwater_and_no_water_is_detected():
    requirement = PlanningRequirement(7, "park", planning_typology="Urban Park")
    update_requirement(requirement, "stormwater management without water")
    result = check_completeness(requirement)
    assert result.conflicts


def test_complete_requirement_generates_ranked_scenarios():
    frame = load_reference_table(str(REFERENCE))
    requirement = PlanningRequirement(
        7, "open park with some shade", "Urban Park", "reduce urban heat",
        "cooling", "scattered_trees", "open", False, ["exclude:Water"],
    )
    proposal = build_complete_proposal(
        requirement, lambda urban_type: get_predictor_values(frame, urban_type)
    )
    assert proposal.scenarios
    assert proposal.scenarios[0].candidate.urban_type == "Scattered trees"
    assert proposal.confidence >= 0.55
    assert "Meteorological" in proposal.external_data_needed[0]
    comparison = explain_scenario_comparison(proposal, "Why not Dense trees?")
    assert "Why Scattered trees instead of Dense trees?" in comparison
    assert "Score difference" in comparison


def test_candidate_score_exposes_components():
    requirement = PlanningRequirement(
        7, "grass park", "Urban Park", "urban greening",
        "recreation", "low_plants", "open", False, ["exclude:Water"],
    )
    candidate = score_requirement(requirement)[0]
    assert candidate.reference_score > 0
    assert candidate.preference_score > 0
    assert candidate.evidence


def test_partial_vector_update_preserves_unchanged_polygon(tmp_path):
    frame = load_reference_table(str(REFERENCE))
    original_values = get_predictor_values(frame, "Compact midrise")
    source = gpd.GeoDataFrame([
        {"fid": 7, **original_values, "geometry": box(0, 0, 1, 1)},
        {"fid": 8, **original_values, "geometry": box(1, 0, 2, 1)},
    ], crs="EPSG:32632")
    source_path = tmp_path / "source.geojson"
    output_path = tmp_path / "updated.geojson"
    source.to_file(source_path, driver="GeoJSON")
    write_attributes_to_vector(
        str(source_path), frame, {7: "Low Plants"}, str(output_path), "fid"
    )
    updated = gpd.read_file(output_path)
    row7 = updated[updated["fid"] == 7].iloc[0]
    row8 = updated[updated["fid"] == 8].iloc[0]
    assert row7["matched_urban_type"] == "Low Plants"
    assert float(row8["TCH"]) == original_values["TCH"]


def test_execution_plan_has_validation_boundary():
    names = [step.name for step in build_execution_plan()]
    assert names[-1] == "validate_outputs"
    assert "write_geojson_attributes" in names


def test_raster_validation_and_replanning(tmp_path):
    path = tmp_path / "TCH_mask.tif"
    metadata = {
        "driver": "GTiff", "height": 2, "width": 2, "count": 1,
        "dtype": "float32", "crs": "EPSG:32632",
        "transform": from_origin(0, 2, 1, 1),
    }
    with rasterio.open(path, "w", **metadata) as dst:
        dst.write(np.ones((2, 2), dtype=np.float32), 1)
    report = validate_raster_outputs({"TCH": str(path)})
    assert report.valid
    assert "downstream LST" in recommend_replanning(report)[0]


def test_invalid_llm_json_falls_back_to_rules():
    result = safe_llm_json("not json", "Polygon 7 should become Water")
    assert result["fallback_used"]
    assert result["intent"] == "set_polygon"


def test_missing_reference_blocks_numerical_generation(tmp_path):
    missing = tmp_path / "missing.csv"
    try:
        require_reference_files(missing)
        assert False, "missing reference should block generation"
    except FileNotFoundError as exc:
        assert "generation is blocked" in str(exc)


def test_failed_action_is_not_reported_as_completed():
    result = safe_domain_action(lambda: 1 / 0)
    assert not result["success"]
    assert "not reported as completed" in result["message"]
