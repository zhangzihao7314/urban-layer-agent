from src.agent.c2_dialogue import (
    continue_c2_proposal,
    extract_predictor_scope,
    start_c2_proposal,
)
from src.layer_alterator_agent.reference_loader import PREDICTOR_COLUMNS
from src.layer_alterator_agent.c2_vector_writer import write_c2_percentage_vector


def test_c2_explicit_natural_language_percentages():
    proposal = start_c2_proposal(
        "Increase trees in Polygon 7 by 30% and reduce grass by 10%", [7, 8]
    )
    assert proposal["status"] == "ready"
    assert proposal["percentages"]["F_TV"] == 30
    assert proposal["percentages"]["TCH"] == 0
    assert proposal["percentages"]["F_G"] == -10
    assert proposal["rule_mode"] == "C3"
    assert set(proposal["active_predictors"]) == {"F_TV", "F_G"}


def test_direct_predictor_percentage_can_include_polygon_between_name_and_value():
    proposal = start_c2_proposal(
        "Increase F_TV in Polygon 7 by 20% and reduce F_G by 10%.", [7, 8]
    )
    assert proposal["status"] == "ready"
    assert proposal["percentages"]["F_TV"] == 20
    assert proposal["percentages"]["F_G"] == -10
    assert proposal["rule_mode"] == "C3"


def test_all_predictors_direct_percentage_is_c2():
    proposal = start_c2_proposal(
        "Increase all predictors in Polygon 7 by 5%.", [7]
    )
    assert proposal["status"] == "ready"
    assert proposal["rule_mode"] == "C2"
    assert set(proposal["active_predictors"]) == set(PREDICTOR_COLUMNS)
    assert set(proposal["percentages"].values()) == {5.0}


def test_c2_asks_for_missing_magnitude_then_completes():
    pending = start_c2_proposal("增加地块7的树木", [7])
    assert pending["status"] == "needs_magnitude"
    proposal = continue_c2_proposal(pending, "增加30%", [7])
    assert proposal["status"] == "ready"
    assert proposal["percentages"]["F_TV"] == 30


def test_c2_polygon_id_ignores_sentence_period():
    pending = start_c2_proposal("Increase trees in Polygon 7.", [7, 8, 9])
    assert pending["status"] == "needs_magnitude"
    assert pending["polygon_id"] == 7
    proposal = continue_c2_proposal(pending, "Increase by 30%.", [7, 8, 9])
    assert proposal["status"] == "ready"
    assert proposal["percentages"]["F_TV"] == 30


def test_percentage_scope_distinguishes_all_and_selected_predictors():
    assert set(extract_predictor_scope("all predictors")) == set(PREDICTOR_COLUMNS)
    assert set(extract_predictor_scope("只修改植被变量")) == {"TCH", "F_G", "F_TV"}


def test_invalid_polygon_reports_available_ids():
    pending = start_c2_proposal("Increase trees in Polygon 99 by 20%", [7, 8, 9])
    assert pending["status"] == "needs_polygon"
    assert "Polygon 99 was not found" in pending["question"]
    assert "7, 8, 9" in pending["question"]


def test_c2_multiple_components_with_sentence_period():
    proposal = start_c2_proposal(
        "Increase water in Polygon 8 by 20% and reduce grass by 10%.",
        [7, 8, 9],
    )
    assert proposal["status"] == "ready"
    assert proposal["polygon_id"] == 8
    assert proposal["percentages"]["F_W"] == 20
    assert proposal["percentages"]["F_G"] == -10


def test_c2_latest_polygon_answer_repairs_saved_pending_state():
    pending = {
        "status": "needs_polygon",
        "original_request": "Increase water in Polygon 8.",
        "question": "Which Polygon?",
    }
    repaired = continue_c2_proposal(pending, "Polygon 8", [7, 8, 9])
    assert repaired["polygon_id"] == 8
    assert repaired["status"] == "needs_magnitude"


def test_c2_percentage_vector_sets_unmentioned_polygons_to_zero(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box

    sample_vector = tmp_path / "source.geojson"
    gpd.GeoDataFrame(
        {"fid": [7, 8]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:3857",
    ).to_file(sample_vector, driver="GeoJSON")
    output = tmp_path / "c2.geojson"
    values = {name: 0.0 for name in ["TCH", "IMD", "BH", "BSF", "SVF", "F_AC", "F_S", "F_M", "F_BS", "F_G", "F_TV", "F_W"]}
    values["F_TV"] = 30
    write_c2_percentage_vector(str(sample_vector), "fid", {7: values}, str(output))
    result = gpd.read_file(output)
    assert result.loc[result["fid"] == 7, "F_TV"].iloc[0] == 30
    assert result.loc[result["fid"] != 7, "F_TV"].eq(0).all()
